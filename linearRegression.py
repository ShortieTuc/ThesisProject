import pickle
import numpy as np
import pandas as pd
import time as t
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


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


start = t.time()

num_lags = 1
df = pd.read_csv('../Thesis/Datasets/Full.csv')
df = create_lags(df, num_lags)
df = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                'netUsageMWh_lag1', 'netUsageMWh'])

# shuffle datapoints
df = df.sample(frac=1)
dataset_arr = df.values
x = dataset_arr[:, :-1]
y = dataset_arr[:, -1:]

# Scale data in [0,1] range
scl = MinMaxScaler()
x = scl.fit_transform(x)
x = np.array(x)
y = scl.fit_transform(y)
y = np.array(y)


# Add the test set
trainset_len = int(len(dataset_arr) - 250)
x_train = x[:trainset_len]
y_train = y[:trainset_len]
x_test = x[trainset_len:]
y_test = y[trainset_len:]

# Fit data into model and make predictions for test set
model = LinearRegression()
model.fit(x_train, y_train)
model = LinearRegression().fit(x_train, y_train)
y_hat = model.predict(x_test)
y_hat = scl.inverse_transform(y_hat)


# model = LinearRegression()
# model.fit(x, y)
# model = LinearRegression().fit(x, y)

# # Save model to disk
# modelname = 'lregr_model.sav'
# pickle.dump(model, open(modelname, 'wb'))

# Metrics for the model
mae = metrics.mean_absolute_error(y_test, y_hat)
mse = metrics.mean_squared_error(y_test, y_hat)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_hat)
end = t.time()
total = end - start
print("Results of metrics:")
print("time in s : ", total)
print("Lags:", num_lags)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)


