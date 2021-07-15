import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as t
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Datasets/Denver.csv')
df = df.query('(hour == 21 or hour == 22 or hour == 23) and day == 2')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.options.display.max_rows
print(df)
