import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define lists
demand = []
timeslot = []
temperature = []
windS = []
windD = []
cloudC = []


# Read bootstrap data from file
f = open("./Games/finals_2020_11_112.xml", "r")
for x in f:
    if "<mwh>" in x:
        demand = x.split(",")
        temp = demand[-1]
        temp = temp.replace('</mwh>\n', '')
        #temp = float(temp) * 1000
        demand[-1] = temp
    if "<weather-report id=" in x:
        content = x.split(" ")
        temp = content[2]
        temp = temp.replace('currentTimeslot=', '')
        temp = temp.replace('"', '')
        timeslot.append(temp)
        temp = content[3]
        temp = temp.replace('temperature=', '')
        temp = temp.replace('"', '')
        temperature.append(temp)
        temp = content[4]
        temp = temp.replace('windSpeed=', '')
        temp = temp.replace('"', '')
        windS.append(temp)
        temp = content[5]
        temp = temp.replace('windDirection=', '')
        temp = temp.replace('"', '')
        windD.append(temp)
        temp = content[6]
        temp = temp.replace('cloudCover=', '')
        temp = temp.replace('"', '')
        temp = temp.replace('/>\n', '')
        cloudC.append(temp)
f.close()

# Read weather report data from file
f = open("./Games/finals_2020_11_112.state", "r")
for x in f:
    if "WeatherReport::" in x:
        content = x.split("::")
        timeslot.append(content[3])
        temperature.append(content[4])
        windS.append(content[5])
        windD.append(content[6])
        str = content[7]
        str = str.replace('\n', '')
        cloudC.append(str)
f.close()

# Read net demand in MWh from file
f = open("./Games/finals_2020_11_112.trace", "r")
for x in f:
    if "DistributionUtilityService: ts" in x:
        content = x.split(",")
        temp = content[2]
        temp = temp.replace(' net = ', '')
        temp = temp.replace('\n', '')
        temp = temp.replace('<mwh>', '')
        demand.append(temp)
f.close()

# Remove excess datapoints
distance = len(timeslot) - len(demand)
for i in range(-distance-1, -1, 1):
     timeslot.remove(timeslot[i])
     temperature.remove(temperature[i])
     windS.remove(windS[i])
     windD.remove(windD[i])
     cloudC.remove(cloudC[i])


# Transform lists to numpy arrays
timeslot = [float(i) for i in timeslot]
timeslot = np.array(timeslot)
temperature = [float(i) for i in temperature]
temperature = np.array(temperature)
windS = [float(i) for i in windS]
windS = np.array(windS)
windD = [float(i) for i in windD]
windD = np.array(windD)
cloudC = [float(i) for i in cloudC]
cloudC = np.array(cloudC)
temp = demand[0]
temp = temp.replace('  <mwh>', '')
demand[0] = temp
demand = [float(i) for i in demand]
demand = np.array(demand)

# create some lag features

lag_timeslot = np.roll(timeslot, 1)
lag_timeslot[0] = 0
lag_temperature = np.roll(temperature, 1)
lag_temperature[0] = 0
lag_windS = np.roll(windS, 1)
lag_windS[0] = 0
lag_windD = np.roll(windD, 1)
lag_windD[0] = 0
lag_cloudC = np.roll(cloudC, 1)
lag_cloudC[0] = 0
lag_demand = np.roll(demand, 1)
lag_demand[0] = 0

# lag_timeslot2 = np.roll(lag_timeslot, 1)
# lag_timeslot2[0] = 0
# lag_temperature2 = np.roll(lag_temperature, 1)
# lag_temperature2[0] = 0
# lag_windS2 = np.roll(lag_windS, 1)
# lag_windS2[0] = 0
# lag_windD2 = np.roll(lag_windD, 1)
# lag_windD2[0] = 0
# lag_cloudC2 = np.roll(lag_cloudC, 1)
# lag_cloudC2[0] = 0
# lag_demand2 = np.roll(lag_demand, 1)
# lag_demand2[0] = 0

# Create dataframe
data_tuples = list(zip(timeslot, temperature, windS, windD, cloudC,
                       lag_timeslot, lag_temperature, lag_windS, lag_windD,
                       lag_cloudC, lag_demand, demand))
df_train = pd.DataFrame(data_tuples, columns=['timeslot', 'temperature', 'windSpeed','windDirection', 'cloudCover',
                                              'lag_timeslot', 'lag_temperature', 'lag_windS',
                                              'lag_windD','lag_cloudC', 'lag_demand', 'mwh'])
print(df_train)
exit()
# df_train = df_train[:(len(demand))]
# df_train.insert(8, "mwh", demand)

# Split dataset to features and target
num_features = 11
dataset_arr = df_train.values
X = dataset_arr[:, :num_features]
y = dataset_arr[:, num_features:]

# Scale data in [0,1] range
scl = MinMaxScaler()
X = scl.fit_transform(X)
X = np.array(X)
y = scl.fit_transform(y)
y = np.array(y)

# Split into train and test data
trainset_len = int(len(dataset_arr) - 168)
X_train = X[:trainset_len]
y_train = y[:trainset_len]
X_test = X[trainset_len:]
y_test = y[trainset_len:]

num_neurons_InL = 24
num_neurons_HL1 = 96
num_neurons_HL2 = 96
num_neurons_OutL = 1

dropout_rate = 0.01
batch_size = 8
epochs = 100

# Build the model
model = Sequential()

model.add(Dense(units=num_neurons_InL, activation='relu', input_shape=X_train[0].shape))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_HL1, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_HL2, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_OutL))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# Fit data to model
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), shuffle=False,
          batch_size=batch_size, verbose=1)

# Save model to file
model.save('FFN_model.h5')

# Optionally, load a model
# model = load_model('FFN_model.h5')

predictions = model.predict(X_test)
predictions = np.array(scl.inverse_transform(predictions.reshape(-1, 1)))
targets = np.array(scl.inverse_transform(y_test.reshape(-1, 1)))

# Visualize results
plot_file_name = 'FFN_EP{}_BS{}.png'.format(epochs, batch_size)
plt.figure(figsize=(18, 10))
plt.plot(predictions, label='Predictions')
plt.plot(targets, label='Target')
plt.legend(loc='best')
plt.title("Epochs = %d HL Neurons = %d Batch Size = %d"%(epochs, num_neurons_HL1, batch_size), fontsize=24)
plt.grid(True)
plt.savefig(plot_file_name)
plt.show()
