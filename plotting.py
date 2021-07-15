import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab


df = pd.read_csv('Results/paper.csv')
#df.insert(0, "Metrics", ['Exec. Time', 'MAE', 'MSE', 'RMSE', 'R^2'], True)
df = df.values.tolist()

plt.title("Linear Regression - Denver Dataset", fontsize=12)
X = np.arange(2)
fig = plt.figure()
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(2, 1, 1)


ax.bar(X + 0.00, df[0], color='darkred', width=0.18, label='Time')
ax.bar(X + 0.25, df[1], color='green', width=0.18, label='MAE')
ax.bar(X + 0.50, df[2], color='purple', width=0.18, label='RMSE')
ax.bar(X + 0.75, df[3], color='lightblue', width=0.18, label='R^2')


ax.set_yscale('logarithmic')
plt.title("Predictors Comparison", fontsize=18)

plt.xticks([r + 0.5 for r in range(len(df[0]))],
           ['Linear Regression', 'NN Regression'], fontsize=15)
# ax.yticks(np.arange(0, 5.1, 0.5))
ax.legend(loc='best')
plt.savefig('predplot.eps', format='eps', bbox_inches = 'tight')
plt.show()
