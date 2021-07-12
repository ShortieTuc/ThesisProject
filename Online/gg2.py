import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler


file = open('./tempFiles/peak.online.json')
data = json.load(file)
df = pd.DataFrame(data, index=[0])
dataset_arr = df.values
num_features = 7
# Read the correct weather data
x = dataset_arr[:, :num_features]
y = dataset_arr[:, num_features:]
x = np.array(x)
y = np.array(y)

print(x, y)