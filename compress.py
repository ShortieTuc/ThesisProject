import pandas as pd


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


df = pd.read_csv('../Thesis/Datasets/Full.csv')
df = create_lags(df, 1)
df = df.filter(['hour', 'day', 'month', 'year', 'temperature', 'windSpeed', 'windDirection', 'cloudCover',
                'netUsageMWh_lag1', 'netUsageMWh'])

lim = [df.min(), df.max()]
lim = pd.DataFrame(lim)
lim.to_csv('./Datasets/lim.csv', index=False)
