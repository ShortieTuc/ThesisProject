from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json
import socket
import os
import shutil


def train(x, y):
    num_neurons_in = 7
    num_neurons_hl1 = 7
    num_neurons_hl2 = 7
    num_neurons_out = 1

    dropout_rate = 0.01
    batch_size = 1
    epochs = 10

    # Build the model
    model = Sequential()

    model.add(Dense(units=num_neurons_in, activation='relu', input_shape=x[0].shape))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_hl1, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_hl2, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_out, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()
    # Fit data to model
    model.fit(x, y, epochs=epochs, shuffle=False, batch_size=batch_size, verbose=2)

    # Save model to file
    model.save('./models/peaks.h5')


def predict(x, model):
    # Make net usage prediction for the next 24 hours
    model = model
    yhat = model.predict(x)
    print(yhat)
    return yhat


def fit(x, y, model):
    # Fit the data to the model
    model = model
    model.fit(x, y, shuffle=False, batch_size=1, verbose=2)
    model.save('./models/peaks.h5')
    return


# TRAINING PHASE
x_scl = MinMaxScaler()
y_scl = MinMaxScaler()
num_features = 7
file = open('tempFiles/peaks.boot.json')
dataset = json.load(file)
dataset = pd.DataFrame(dataset)
dataset['peak'] = np.where(dataset['peak'] == True, 1.0, 0.0)
# Split dataset to features and target
dataset_arr = dataset.values
x = dataset_arr[:, :num_features]
x = x_scl.fit_transform(x)
x = np.array(x).astype('float32')
y = dataset_arr[:, num_features+1:]
y = y_scl.fit_transform(y)
y = np.array(y).astype('float32')
file.close()
train(x, y)

# PREDICTING PHASE
yhat = []
file = open('./tempFiles/forecast24.online.json')
forecast = json.load(file)
x = pd.DataFrame(forecast)
x = x_scl.fit_transform(x)
x = np.array(x).astype('float32')
file.close()
model = load_model('models/peaks.h5')
yhat = predict(x, model)
yhat = y_scl.inverse_transform(yhat)
yhat = [1.0 if yhat[i] > 0.5 else 0.0 for i in range(len(yhat))]
fit(x, yhat, model)

# FITTING PHASE
model = load_model('models/peaks.h5')
file = open('./tempFiles/peak.online.json')
fit_data = json.load(file)
file.close()
df = pd.DataFrame(fit_data, index=[0])
correct = df.values
df['peak'] = np.where(df['peak'] == 'True', 1.0, 0.0)
dataset = dataset.append(df, ignore_index=True)
num_features = 7
# Read the correct weather data
x = correct[:, :num_features]
y = correct[:, num_features + 1:]
fit(x, y, model)

# RETRAINING PHASE
threshold = 40
# peaks values correction
dataset['peak'] = np.where(dataset['netUsageMWh'] > threshold, 1.0, 0.0)
dataset_arr = dataset.values
x = dataset_arr[:, :num_features]
x = x_scl.fit_transform(x)
x = np.array(x)
y = dataset_arr[:, num_features+1:]
y = y_scl.fit_transform(y)
y = np.array(y)
train(x, y)


