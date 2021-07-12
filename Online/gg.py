from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np
import json
import time
import socket
import os
import shutil


def bootstrap_edit():
    func_start_time = time.time()
    # read weather bootstrap data file and make a data frame of it
    file = open('tempFiles/weather.boot2.json')
    data = json.load(file)
    x = pd.DataFrame(data)
    file.close()
    # read customer net usage file and split it into data frames per customer
    file = open('tempFiles/customers.boot.json')
    data = json.load(file)
    file.close()
    y = pd.DataFrame(data)
    y = y.drop('powerType', 1)
    y = y.explode('netUsage')
    splits = list(y.groupby("customerName"))
    # iteratively train customers' models
    cust_cnt = 0
    cust_names = []
    times_record = []
    for i in range(0, 2):
        # for i in range(len(splits)):
        for j in range(1, len(splits[0])):
            temp_y = splits[i][j]
            name = temp_y.iloc[0]['customerName']
            temp_y = temp_y.drop('customerName', 1)

            start_time = time.time()
            train_model(x, temp_y, name)
            cust_names.append(name)
            cust_cnt += 1
            elapsed_time = time.time() - start_time
            times_record.append(elapsed_time)
            print("Training time for customer " + str(cust_cnt) + ": " + str(elapsed_time))

    times_record = np.array(times_record)
    print("AVG Training time: " + str(np.average(times_record)))
    print("MAX Training time: " + str(np.max(times_record)) + "(CustID: " + str(np.argmax(times_record) + 1) + ")")
    print("MIN Training time: " + str(np.min(times_record)) + "(CustID: " + str(np.argmin(times_record) + 1) + ")")
    func_elapsed_time = time.time() - func_start_time
    print("Total script time: " + str(func_elapsed_time / 60) + " mins")

    return cust_names


# This function trains the customer's model
def train_model(x, y, name):
    file_name = './models/{}.h5'.format(name)

    scl = MinMaxScaler(feature_range=(-1, 1))
    x = np.array(x)
    x = scl.fit_transform(x)
    y = np.array(y)
    y = scl.fit_transform(y)

    num_neurons_in = 4
    num_neurons_hl1 = 7
    num_neurons_hl2 = 7
    num_neurons_out = 1

    dropout_rate = 0.01
    # TODO check in sim, laptop
    batch_size = 4
    epochs = 10

    # Build the model
    model = Sequential()

    model.add(Dense(units=num_neurons_in, activation='relu', input_shape=x[0].shape))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_hl1, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_hl2, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_neurons_out))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # model.summary()

    # Fit data to model
    model.fit(x, y, epochs=epochs, shuffle=False, batch_size=batch_size, verbose=0)

    # Save model to file
    model.save(file_name)


# This function is used to predict the customer's net usage for future time slots
def predict_customer(forecast, model):
    # read weather forecast data file and make a data frame of it

    x = pd.DataFrame(forecast)
    x = np.array(x)

    scl = MinMaxScaler(feature_range=(-1, 1))
    x = scl.fit_transform(x)

    model = model
    y = model.predict(x)
    y = [element * 100 for element in y]
    y = np.array(y)

    return y


def load_models(customers):
    models = []
    for i in range(len(customers)):
        file_name = './models/{}.h5'.format(customers[i])
        models.append(file_name)
        models[i] = load_model(models[i])
    return models


# def fit_models(model, x, y):
#     model = model
#     model.fit(self, x, y, shuffle=False, batch_size=4, verbose=0)
#     model.save(model)


# num_cores = 2
net_usage = np.ndarray((24,), float)
file = open('./tempFiles/forecast24.online.json')
forecast = json.load(file)
customers = bootstrap_edit()
# inputs = range(len(customers))
models = load_models(customers)


# Parallel(n_jobs=num_cores)(delayed(predict_prep(customers))(i) for i in inputs)

# Predict customer usage for the next 24 time slots and summarize for all customers to get total net_usage
for i in range(len(models)):
    customer_usage = predict_customer(forecast, models[i])
    for j in range(len(customer_usage)):
        net_usage[j] += customer_usage[j]
print(net_usage)
# correct_x = open('./tempFiles/forecast24.online.json')
# correct_y = open('./tempFiles/forecast24.online.json')
# for i in range(len(models)):
#     fit_models(correct_x, correct_y, models[i])
