import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from statistics import mean


def plot_per_game(regr, clf):

    for i in range(1, 16):

        game_diff = regr.loc[regr['Game_Num'] == i, 'Game_Diff']
        game_diff = game_diff.values.tolist()
        regr_train_time = regr.loc[regr['Game_Num'] == i, 'Train_Time']
        regr_train_time = regr_train_time.values.tolist()
        regr_pred_time = regr.loc[regr['Game_Num'] == i, 'Pred_Time']
        regr_pred_time = regr_pred_time.values.tolist()
        lags = regr.loc[regr['Game_Num'] == i, 'Lags']
        lags = lags.values.tolist()
        mse = regr.loc[regr['Game_Num'] == i, 'MSE']
        mse = mse.values.tolist()
        mae = regr.loc[regr['Game_Num'] == i, 'MAE']
        mae = mae.values.tolist()
        rmse = regr.loc[regr['Game_Num'] == i, 'RMSE']
        temp = rmse.values.tolist()
        rmse = [math.sqrt(x) for x in temp]
        r2 = regr.loc[regr['Game_Num'] == i, 'R^2']
        r2 = r2.values.tolist()

        clf_train_time = clf.loc[clf['Game_Num'] == i, 'Train_Time']
        clf_train_time = clf_train_time.values.tolist()
        clf_pred_time = clf.loc[clf['Game_Num'] == i, 'Pred_Time']
        clf_pred_time = clf_pred_time.values.tolist()
        acc = clf.loc[clf['Game_Num'] == i, 'ACC']
        acc = acc.values.tolist()
        prec = clf.loc[clf['Game_Num'] == i, 'PREC']
        prec = prec.values.tolist()
        rec = clf.loc[clf['Game_Num'] == i, 'REC']
        rec = rec.values.tolist()
        f1 = clf.loc[clf['Game_Num'] == i, 'F1']
        f1 = f1.values.tolist()

        plot1 = 'Plots/Regr_Game{}.png'.format(i)

        # set width of bar
        barWidth = 0.10
        fig = plt.subplots(figsize=(6, 6))

        # Set position of bar on X axis
        br1 = np.arange(len(lags))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        plt.bar(br1, mse, color='r', width=barWidth,
                edgecolor='grey', label='MSE')
        plt.bar(br2, mae, color='g', width=barWidth,
                edgecolor='grey', label='MAE')
        plt.bar(br3, rmse, color='b', width=barWidth,
                edgecolor='grey', label='RMSE')
        plt.bar(br4, r2, color='y', width=barWidth,
                edgecolor='grey', label='R^2')

        plt.title("Game #{} with Regression".format(i), fontsize=24)
        plt.xlabel('Number of Lags', fontweight='bold')
        # plt.ylabel('Regression', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(lags))],
                   ['3', '5', '10'])
        plt.ylim(-1, 1)
        plt.legend(loc='best')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig(plot1, dpi=100)

        plot2 = 'Plots/Clf_Game{}.png'.format(i)
        # set width of bar
        barWidth = 0.10
        fig = plt.subplots(figsize=(6, 6))

        # Set position of bar on X axis
        br1 = np.arange(len(lags))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        plt.bar(br1, acc, color='r', width=barWidth,
                edgecolor='grey', label='ACC')
        plt.bar(br2, prec, color='g', width=barWidth,
                edgecolor='grey', label='PREC')
        plt.bar(br3, rec, color='b', width=barWidth,
                edgecolor='grey', label='REC')
        plt.bar(br4, f1, color='y', width=barWidth,
                edgecolor='grey', label='F1')

        plt.title("Game #{} with Classification".format(i), fontsize=24)
        plt.xlabel('Number of Lags', fontweight='bold')
        # plt.ylabel('Regression', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(lags))],
                   ['3', '5', '10'])
        plt.ylim(0, 2)
        plt.legend(loc='best')
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig(plot2, dpi=100)


def plot_average_regr(mse, mae, rmse, r2):

    plot = 'Plots/Regration_Average.png'
    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize=(6, 6))

    # Set position of bar on X axis
    br1 = np.arange(len(mse))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, mse, color='r', width=barWidth,
            edgecolor='grey', label='MSE')
    plt.bar(br2, mae, color='g', width=barWidth,
            edgecolor='grey', label='MAE')
    plt.bar(br3, rmse, color='b', width=barWidth,
            edgecolor='grey', label='RMSE')
    plt.bar(br4, r2, color='y', width=barWidth,
            edgecolor='grey', label='R^2')

    plt.title("Regression Average Metrics", fontsize=24)
    plt.xlabel('Number of Lags', fontweight='bold')
    # plt.ylabel('Regression', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(mse))],
               ['3', '5', '10'])
    plt.ylim(-1, 1)
    plt.legend(loc='best')
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig(plot, dpi=100)


def plot_average_clf(acc, prec, rec, f1):
    plot = 'Plots/Classification_Average.png'
    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize=(6, 6))

    # Set position of bar on X axis
    br1 = np.arange(len(acc))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, acc, color='r', width=barWidth,
            edgecolor='grey', label='ACC')
    plt.bar(br2, prec, color='g', width=barWidth,
            edgecolor='grey', label='PREC')
    plt.bar(br3, rec, color='b', width=barWidth,
            edgecolor='grey', label='REC')
    plt.bar(br4, f1, color='y', width=barWidth,
            edgecolor='grey', label='F1')

    plt.title("Classification Average Metrics", fontsize=24)
    plt.xlabel('Number of Lags', fontweight='bold')
    # plt.ylabel('Regression', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(acc))],
               ['3', '5', '10'])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 2, 0.05))
    plt.legend(loc='best')
    fig3 = plt.gcf()
    plt.show()
    fig3.savefig(plot, dpi=100)


# Read csv records
regr = pd.read_csv('Datasets/regression.csv')
clf = pd.read_csv('classification.csv')

# Regression metric according to number of lag features
mseav = [mean(regr.loc[regr['Lags'] == 3, 'MSE']),
         mean(regr.loc[regr['Lags'] == 5, 'MSE']),
         mean(regr.loc[regr['Lags'] == 10, 'MSE'])]
maeav = [mean(regr.loc[regr['Lags'] == 3, 'MAE']),
         mean(regr.loc[regr['Lags'] == 5, 'MAE']),
         mean(regr.loc[regr['Lags'] == 10, 'MAE'])]
rmseav = [mean(regr.loc[regr['Lags'] == 3, 'RMSE']),
          mean(regr.loc[regr['Lags'] == 5, 'RMSE']),
          mean(regr.loc[regr['Lags'] == 10, 'RMSE'])]
r2av = [mean(regr.loc[regr['Lags'] == 3, 'R^2']),
        mean(regr.loc[regr['Lags'] == 5, 'R^2']),
        mean(regr.loc[regr['Lags'] == 10, 'R^2'])]


# Classification metrics according to number of lag features
accav = [mean(clf.loc[regr['Lags'] == 3, 'ACC']),
         mean(clf.loc[regr['Lags'] == 5, 'ACC']),
         mean(clf.loc[regr['Lags'] == 10, 'ACC'])]
precav = [mean(clf.loc[regr['Lags'] == 3, 'PREC']),
          mean(clf.loc[regr['Lags'] == 5, 'PREC']),
          mean(clf.loc[regr['Lags'] == 10, 'PREC'])]
recav = [mean(clf.loc[regr['Lags'] == 3, 'REC']),
         mean(clf.loc[regr['Lags'] == 5, 'REC']),
         mean(clf.loc[regr['Lags'] == 10, 'REC'])]
f1av = [mean(clf.loc[regr['Lags'] == 3, 'F1']),
        mean(clf.loc[regr['Lags'] == 5, 'F1']),
        mean(clf.loc[regr['Lags'] == 10, 'F1'])]

#plot_per_game(regr, clf)
#plot_average_regr(mseav, maeav, rmseav, r2av)
plot_average_clf(accav, precav, recav, f1av)