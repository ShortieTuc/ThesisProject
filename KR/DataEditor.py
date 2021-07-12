from KR import GKR
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

num_features = 4

# Read the CSV input file and show first 5 rows
# 230 days * 24 hours = 5520 datapoints
df_train = pd.read_csv('../Datasets/Better.csv')
df_train = df_train.drop(['Day', 'Hour'], axis='columns')
df_train.head()

# Split dataset to features and target
dataset_arr = df_train.values
X = dataset_arr[:, :num_features]
Y = dataset_arr[:, num_features:]
X = np.array(X)
Y = np.array(Y)

# Check array shapes
print('X: ', X.shape)
print('Y: ', Y.shape)

# Scale data in [0,1]
scl = MinMaxScaler()
X = scl.fit_transform(X)
X = np.array(X)
Y = scl.fit_transform(Y)
Y = np.array(Y)

# compute the standard deviation of Y values
std = np.std(X)

# Create the kernels
gkr = GKR.GKR(X, Y, 1)

# Give the feature values for the prediction and scale back
prediction = gkr.predict(X[-1])
prediction = scl.inverse_transform(prediction.reshape(1, -1))
print(prediction)



