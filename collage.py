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


def collage():

    games = ['DM', 'Phoenix']

    game_name = './Datasets/' + games[0] + '.csv'
    df = pd.read_csv(game_name)

    for i in range(1, len(games)):
        game_name = './Datasets/' + games[i] + '.csv'

        temp_df = pd.read_csv(game_name)
        df = df.append(temp_df)

    return df


df = collage()
df.to_csv('./Datasets/Full.csv', index=False)

