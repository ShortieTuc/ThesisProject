import pandas as pd
import matplotlib.pyplot as plt


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


num_lags = 1
df = pd.read_csv('../Thesis/Datasets/Phoenix.csv')
df = create_lags(df, num_lags)
df = df.filter(['hour', 'day', 'month', 'year', 'tmpr', 'windSpd', 'windDir', 'cloudCov', 'mwh'])
df.to_csv('./Datasets/Phoenix.csv', index=False)
print(df)

# plt.figure()
# a = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10,10))
# plt.show()


