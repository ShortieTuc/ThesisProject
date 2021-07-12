from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json
import socket
import os
import shutil


def train(x, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x, y.ravel())
    filename = 'models/peaks.h5'
    pickle.dump(model, open(filename, 'wb'))

def predict(x, model):
    # Make net usage prediction for the next 24 hours
    model = model
    yhat = model.predict(x)
    return yhat


def fit(x, y, model):
    # Fit the data to the model
    model = model
    model.fit(x, y.ravel())
    filename = 'models/peaks.h5'
    pickle.dump(model, open(filename, 'wb'))
    return

PORT = 8098
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print(sock)

server_address = ('localhost', PORT)
sock.bind(server_address)
sock.listen(1)
c = 0

# TODO disable flags
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
        y = dataset_arr[:, num_features + 1:]
        y = y_scl.fit_transform(y)
        y = np.array(y).astype('float32')
        file.close()

        trained = True
        out = "ok\n"
        connection.sendall(out.encode('utf-8'))
        train(x, y)
    elif data == "predict":
        if not trained:
            out = "error\n"
            connection.sendall(out.encode('utf-8'))
        else:
            # PREDICTING PHASE
            yhat = []
            file = open('./tempFiles/forecast24.online.json')
            forecast = json.load(file)
            x = pd.DataFrame(forecast)
            x = x_scl.fit_transform(x)
            x = np.array(x).astype('float32')
            file.close()
            filename = 'models/peaks.h5'
            model = pickle.load(open(filename, 'rb'))
            yhat = predict(x, model)
            # yhat = y_scl.inverse_transform(yhat)
            yhat = [1.0 if yhat[i] > 0.5 else 0.0 for i in range(len(yhat))]
            y = np.array(y).astype('float32')

            out = ' '.join([str(elem) for elem in np.array(yhat).flatten()])
            print("Predictions:")
            print(out)
            out = out + "\nok\n"
            connection.sendall(out.encode('utf-8'))
    elif data == "fit":
        filename = 'models/peaks.h5'
        model = pickle.load(open(filename, 'rb'))
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
        x = x_scl.fit_transform(x)
        x = np.array(x).astype('float32')
        y = correct[:, num_features + 1:]
        y = y_scl.fit_transform(y)
        y = np.array(y).astype('float32')
        print(y)
        fit(x, y, model)

        out = "ok\n"
        connection.sendall(out.encode('utf-8'))
    elif data == "retrain":
        print("retrain")
        print(whole_data.split()[1])
        threshold = int(whole_data.split()[1])
        print(type(threshold))
        # RETRAINING PHASE
        threshold = 40
        # peaks values correction
        dataset['peak'] = np.where(dataset['netUsageMWh'] > threshold, 1.0, 0.0)
        dataset_arr = dataset.values
        x = dataset_arr[:, :num_features]
        x = x_scl.fit_transform(x)
        x = np.array(x).astype('float32')
        y = dataset_arr[:, num_features + 1:]
        y = y_scl.fit_transform(y)
        y = np.array(y).astype('float32')
        train(x, y)

        out = "ok\n"
        connection.sendall(out.encode('utf-8'))
        train(x, y)

    elif data == "reset":
        print("reset..")
        trained = False
        if os.path.exists("models"):
            shutil.rmtree('models')

        os.mkdir("models")
        y = "ok\n"
        connection.sendall(y.encode('utf-8'))
    else:
        print("error")
    c = c + 1
