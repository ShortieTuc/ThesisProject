import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import sys
import time as t


# Create lag features
def create_lags(dataset, num):
    new_dict = {}
    for col_name in dataset:
        new_dict[col_name] = dataset[col_name]
        # create lagged Series
        for l in range(1, int(num) + 1):
            new_dict['%s_lag%d' % (col_name, l)] = dataset[col_name].shift(l)
    res = pd.DataFrame(new_dict, index=dataset.index)
    res = res.fillna(0)
    return res


def processData(x, y, mins_look_back, forward_mins, jump=1):
    X, Y = [], []
    for i in range(0, len(x) - mins_look_back - forward_mins + 1, jump):
        X.append(x[i:(i + mins_look_back)])
        Y.append(y[(i + mins_look_back):(i + mins_look_back + forward_mins)])
    return np.array(X), np.array(Y)


def processDataTest(x, mins_look_back, jump=1):
    X = []
    for i in range(0, len(x) - mins_look_back + 1, jump):
        X.append(x[i:(i + mins_look_back)])
    return np.array(X)

argv = sys.argv[1:]

# Define inputs and output length
start = t.time()
mins_look_back = 24
forward_mins = 24
num_features = 7
num_lags = 1


# Read the CSV input file and show first 5 rows
df = pd.read_csv('../Thesis/Datasets/Denver.csv')
df_train = create_lags(df, num_lags)
df_train = df_train.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                'netUsageMWh_lag1', 'netUsageMWh'])

# Split dataset to features and target
dataset_arr = df_train.values
X = dataset_arr[:, :num_features]
y = dataset_arr[:, num_features - 1:]

# Scale data in [0,1] range
scl = MinMaxScaler()
X = scl.fit_transform(X)
X = np.array(X)
y = scl.fit_transform(y)
y = np.array(y)

# Split into train and test data
trainset_len = int(len(dataset_arr) - 2400)
X_data_train = X[:trainset_len]
y_data_train = y[:trainset_len]
X_data_test = X[trainset_len:]
y_test = y[trainset_len:]

# Make data fit-able to RNNs
X_train, y_train = processData(X_data_train, y_data_train, mins_look_back, forward_mins)
X_test = processDataTest(X_data_test, mins_look_back)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Check array shapes
print('X Train: ', X_train.shape)
print('y Train: ', y_train.shape)
print('X Test: ', X_test.shape)
print('y Test: ', y_test.shape)

num_neurons_L1 = 24
num_neurons_L2 = 24
num_neurons_L3 = 24
num_neurons_L4 = 24
dropout_rate = 0.01
epochs = 10
batch_size = 8

# Build the model
model = Sequential()

model.add(LSTM(units=num_neurons_L1, input_shape=(mins_look_back, num_features), return_sequences=True))
model.add(Dropout(rate=dropout_rate))

model.add(LSTM(units=num_neurons_L2, return_sequences=True))
model.add(Dropout(rate=dropout_rate))

model.add(LSTM(units=num_neurons_L3, return_sequences=True))
model.add(Dropout(rate=dropout_rate))

model.add(LSTM(units=num_neurons_L4))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(forward_mins))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# Fit data to model
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate),
          shuffle=False, batch_size=batch_size, verbose=1)

# Save model to file
file_name = 'LSTM_EP{}_BS{}.h5'.format(epochs, batch_size)
model.save(file_name)

# Optionally, load a model
# model = load_model('LSTM_model.h5')

# Make some predictions
y_hat = model.predict(X_test)
y_hat = np.array(scl.inverse_transform(y_hat.reshape(-1, 1)))
y_test = np.array(scl.inverse_transform(y_test.reshape(-1, 1)))

# Metrics for the model
mae = metrics.mean_absolute_error(y_test, y_hat)
mse = metrics.mean_squared_error(y_test, y_hat)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_hat)
end = t.time()
total = end - start
print("Results of metrics:")
print("time in s : ", total)
# print("Lags:", num_lags)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
# # Visualize results
# plot_file_name = 'PREDS_LSTM_EP{}_BS{}.png'.format(epochs, batch_size)
# plt.figure(figsize=(18, 10))
# plt.plot(predictions, label='Predictions')
# plt.plot(targets, label='Target')
# plt.legend(loc='best')
# plt.grid(True)
# plt.savefig(plot_file_name)
# # plt.show()
