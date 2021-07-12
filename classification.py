import pandas as pd
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics

# Split dataset to features and target
df = pd.read_csv('Datasets/Denver.csv')
dataset_arr = df.values
X = dataset_arr[:, :-1]
y = dataset_arr[:, -1:]


# Scale data in [0,1] range
scl = MinMaxScaler()
X = scl.fit_transform(X)
X = np.array(X)
y = scl.fit_transform(y)
y = np.array(y)

# Split into train and test data
trainset_len = int(len(dataset_arr) - 672)
X_train = X[:trainset_len]
y_train = y[:trainset_len]
X_test = X[trainset_len:]
y_test = y[trainset_len:]

num_neurons_InL = 24
num_neurons_HL1 = 24
num_neurons_HL2 = 24
num_neurons_OutL = 1

dropout_rate = 0.01
batch_size = 4
epochs = 50

# model = load_model('clf.h5')
# model.fit(X,y)
# model.save('clf.h5')

# Build the model
train0 = time.time()
model = Sequential()

model.add(Dense(units=num_neurons_InL, activation='relu', input_shape=X_train[0].shape))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_HL1, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_HL2, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(units=num_neurons_OutL, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# Fit data to model
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), shuffle=False,
          batch_size=batch_size, verbose=1)

# Save model to file
model.save('clf.h5')

# Optionally, load a model
# model = load_model('clf.h5')

train1 = time.time()

pred0 = time.time()
predictions = model.predict(X_test)
predictions = np.array(scl.inverse_transform(predictions.reshape(-1, 1)))
predictions = [1.0 if predictions[i] > 0.4 else 0.0 for i in range(len(predictions))]
targets = np.array(scl.inverse_transform(y_test.reshape(-1, 1)))
targets = targets.tolist()

d = {'targets': targets, 'predictions': predictions}
df2 = pd.DataFrame(d)
csv_name = './10/HL{}_Ep{}.csv'.format(num_neurons_HL1, epochs)
df2.to_csv(csv_name, index=False)

acc = sklearn.metrics.accuracy_score(y_test, predictions)
print("ACC: ", acc)
prec = sklearn.metrics.precision_score(y_test, predictions, average='weighted')
print("PREC: ", prec)
rec = sklearn.metrics.recall_score(y_test, predictions, average='weighted')
print("REC: ", rec)
f1 = sklearn.metrics.f1_score(y_test, predictions, average='weighted')
print("F1: ", f1)
pred1 = time.time()
train_time = train1 - train0
pred_time = pred1 - pred0

# with open('classification.csv', 'a+', newline='') as write_obj:
#     writer = csv.writer(write_obj)
#     writer.writerow([parse.game_num, parse.peaks, train_time, pred_time, parse.lags_num, acc, prec, rec, f1])

# Visualize results
plot_file_name = 'Clf_Ep{}_B{}.png'.format(epochs, batch_size)
plt.figure(figsize=(18, 10))
plt.plot(predictions, label='Predictions')
plt.plot(targets, label='Target')
plt.legend(loc='best')
plt.title("Epochs = %d HL Neurons = %d Batch Size = %d M.S.E. = %.3f M.A.E. = %.3f"
          % (epochs, num_neurons_HL1, batch_size, acc, prec), fontsize=20)
plt.grid(True)
plt.savefig(plot_file_name)
plt.show()
