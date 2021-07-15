import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Results/ff.csv')

plt.figure(figsize=(9, 8))
plots = sns.barplot(x="name", y="value", data=df)

# Iterrating over the bars one-by-one
for bar in plots.patches:
    # Using Matplotlib's annotate function and
    # passing the coordinates where the annotation shall be done
    # x-coordinate: bar.get_x() + bar.get_width() / 2
    # y-coordinate: bar.get_height()
    # free space to be left to make graph pleasing: (0, 8)
    # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

# Setting the x-axis label and its size
plt.xlabel("All Features - All Locations - No Lags", size=15)

plots.set_yscale('log')
plt.title("Feed Forward Neural Networks", fontsize=18)
# Setting the y-axis label and its size
plt.ylabel("log", size=15)

# Plotting the graph
plt.show()





# # df.insert(0, "Metrics", ['Exec. Time', 'MAE', 'MSE', 'RMSE', 'R^2'], True)
# df = df.values.tolist()
#
# X = np.arange(2)
# fig = plt.figure()
# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(2, 1, 1)
#
# # ax.bar(X + 0.00, df[0], color='darkred', width=0.18, label='Time')
#
# ax.bar(X + 0.25, df[0], color='darkgreen', width=0.1, label='time(s)')
# ax.bar(X + 0.5, df[1], color='purple', width=0.1, label='MAE')
# ax.bar(X + 0.75, df[2], color='lightblue', width=0.1, label='RMSE')
# # ax.bar(X + 1, df[3], color='darkred', width=0.1, label='R^2')
#
# ax.set_yscale('log')
# plt.title('Feed Forward Neural Networks', fontsize=18)
# plt.xticks([1],
#            ['Dataset with 1 Lag Feature and no Wind Features'], fontsize=15)
#
# # ax.yticks(np.arange(0, 5.1, 0.5))
# ax.legend(loc='best')
# plt.savefig('ff1.eps', format='eps', bbox_inches = 'tight')
# plt.show()
