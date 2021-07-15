import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import json
import socket
import os
import shutil


# Create lag features
def create_lags(dataset, num):
    new_dict = {}
    for col_name in dataset:
        new_dict[col_name] = dataset[col_name]
        # create lagged Series
        for k in range(1, int(num) + 1):
            new_dict['%s_lag%d' % (col_name, k)] = dataset[col_name].shift(k)
    res = pd.DataFrame(new_dict, index=dataset.index)
    res = res.fillna(0)
    return res


# Useless for now, but you never know
# Read the bootstrap data, convert them to dataset and fit them to the model
def boot_train(lim):
    # load trained model from disk
    lrmodel = pickle.load(open('lregr_model.sav', 'rb'))
    # read bootstrap data and make it a dataset
    file = open('../Thesis/tempFiles/peaks.boot.json')
    df = json.load(file)
    df = pd.DataFrame(df)
    file.close()
    # create lag and split dataset into features and target
    lags_num = 1
    df = create_lags(df, lags_num)
    df = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                    'netUsageMWh_lag1', 'netUsageMWh'])
    x = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                   'netUsageMWh_lag1'])
    y = df.filter(['netUsageMWh'], axis=1)
    recent = y['netUsageMWh'].iloc[-1]
    # Scale data in [0,1] range
    scl = MinMaxScaler()
    x = scl.fit_transform(x)
    x = np.array(x)
    y = scl.fit_transform(y)
    y = np.array(y)
    # fit bootstrap data to the model
    lrmodel.fit(x, y)
    lrmodel = LinearRegression().fit(x, y)
    # Save model to disk
    modelname = 'lregr_model.sav'
    pickle.dump(lrmodel, open(modelname, 'wb'))
    return recent


# Find the upper and lesser limit values in the dataset
def initialization():
    df = pd.read_csv('../Thesis/Datasets/lim.csv')
    df = create_lags(df, 1)
    df = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                    'netUsageMWh_lag1', 'netUsageMWh'])
    file = open('../Thesis/tempFiles/peaks.boot.json')
    df2 = json.load(file)
    df2 = pd.DataFrame(df2)
    file.close()
    df2 = create_lags(df2, 1)
    df2 = df2.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                      'netUsageMWh_lag1', 'netUsageMWh'])
    lim = [df.min(), df.max(), df2.iloc[-1]]
    lim = np.array(lim)
    lim = pd.DataFrame(lim, columns=['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection',
                                     'cloudCover', 'netUsageMWh_lag1', 'netUsageMWh'])
    return lim


# Read the datapoints, predict the target value for each of them
def predict8(lim):
    predictions = []
    file = open('../Thesis/tempFiles/forecast8.online.json')
    df = json.load(file)
    df = pd.DataFrame(df)
    df = create_lags(df, 1)
    file.close()
    df = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                    'netUsageMWh_lag1'])
    for i in range(8):
        datapoint = df.iloc[i]
        prediction = predict1(datapoint, lim)
        predictions = np.append(predictions, prediction)
        lim = lim.append(datapoint)
        lim['netUsageMWh_lag1'].iloc[-1] = lim['netUsageMWh_lag1'].iloc[-2]
        lim['netUsageMWh'].iloc[-1] = prediction
    return predictions


# Make prediction for a single datapoint
def predict1(datapoint, lim):
    limx = lim.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                       'netUsageMWh_lag1'])
    limy = lim.filter(['netUsageMWh'])
    limx = limx.append(datapoint)
    limx['netUsageMWh_lag1'].iloc[-1] = limy.iloc[-1]
    scl = MinMaxScaler()
    limx = scl.fit_transform(limx)
    limy = scl.fit_transform(limy)
    limx = np.array(limx)
    limy = np.array(limy)
    # load trained model from disk
    lrmodel = pickle.load(open('lregr_model.sav', 'rb'))
    preds = lrmodel.predict(limx)
    preds = preds[-1]
    limy = np.append(limy, preds)
    limy = [limy]
    limy = scl.inverse_transform(limy)
    limy = limy[-1]
    return limy[-1]


PORT = 8098
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print(sock)

server_address = ('localhost', PORT)
sock.bind(server_address)
sock.listen(1)
c = 0

trained = False
while True:

    print('\nwaiting for a connection\n')

    # wait for client input
    connection, client_address = sock.accept()
    data = connection.recv(16)
    data = data.decode('ascii')
    print(data)
    whole_data = data.rstrip()
    data = data.rstrip().split()[0]

    if data == "boot_train":
        lim = initialization()
        trained = True
        out = "ok\n"
        connection.sendall(out.encode('utf-8'))

    elif data == "predict":
        if not trained:
            out = "error\n"
            connection.sendall(out.encode('utf-8'))
        else:
            predictions = predict8(lim)
            out = ' '.join([str(elem) for elem in np.array(predictions).flatten()])
            print("Predictions:")
            print(out)
            out = out + "\nok\n"
            connection.sendall(out.encode('utf-8'))
    elif data == "reset":
        print("reset..")
        flag = False
        if os.path.exists("models"):
            shutil.rmtree('models')

        os.mkdir("models")
        y = "ok\n"
        connection.sendall(y.encode('utf-8'))
    else:
        print("error")
    c = c + 1


