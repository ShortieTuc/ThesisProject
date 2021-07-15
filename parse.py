import csv
import numpy as np
import pandas as pd


# Threshold calculator
def calc_threshold(list):
    mean = np.mean(list)
    dev = np.std(list)
    threshold = mean + 1.25 * dev
    return threshold


# Parse input
game_num = '376'
lags_num = '15'
game_name = './Games/finals_2020_11_' + game_num
xml = game_name + '.xml'
state = game_name + '.state'
trace = game_name + '.trace'

# Define lists
demand = []
timeslot = []
temperature = []
windS = []
windD = []
cloudC = []
threshold = []
peak = []

# Read bootstrap data from file
f = open(xml, "r")
for x in f:
    if "<mwh>" in x:
        demand = x.split(",")
        temp = demand[-1]
        temp = temp.replace('</mwh>\n', '')
        demand[-1] = temp

    if "<weather-report id=" in x:
        content = x.split(" ")
        temp = content[2]
        temp = temp.replace('currentTimeslot=', '')
        temp = temp.replace('"', '')
        timeslot.append(temp)
        temp = content[3]
        temp = temp.replace('temperature=', '')
        temp = temp.replace('"', '')
        temperature.append(temp)
        temp = content[4]
        temp = temp.replace('windSpeed=', '')
        temp = temp.replace('"', '')
        windS.append(temp)
        temp = content[5]
        temp = temp.replace('windDirection=', '')
        temp = temp.replace('"', '')
        windD.append(temp)
        temp = content[6]
        temp = temp.replace('cloudCover=', '')
        temp = temp.replace('"', '')
        temp = temp.replace('/>\n', '')
        cloudC.append(temp)
f.close()

# Read weather report data from file
pred = 0
f = open(state, "r")
for x in f:
    if "WeatherReport::" in x:
        content = x.split("::")
        timeslot.append(content[3])
    if "WeatherForecastPrediction::" in x:
        if pred > 0:
            pred -= 1
        if pred == 0:
            pred = 24
            content = x.split("::")
            temperature.append(content[4])
            windS.append(content[5])
            windD.append(content[6])
            str = content[7]
            str = str.replace('\n', '')
            cloudC.append(str)
f.close()

# Read net demand in MWh from file
f = open(trace, "r")
flag = 0
for x in f:
    if "server.CompetitionControlService: createInitialTimeslots" in x:
        if flag == 0:
            flag = 1
            content = x.split(",")
            content = content[3].split("-")
            temp = content[0]
            temp = temp.replace(' at ', '')
            init_year = temp
            init_month = content[1]
            init_month = int(init_month)
            temp = content[2]
            day = temp[:2]
            init_day = int(day)

            hour = [None] * len(timeslot)
            dayofweek = [None] * len(timeslot)
            dayofmonth = [int(init_day)] * len(timeslot)
            month = [int(init_month)] * len(timeslot)
            year = [int(init_year)] * len(timeslot)
            hour[0] = 0
            dayofweek[0] = 7

        # hours
        for i in range(1, len(timeslot)):
            if hour[i-1] == 23:
                hour[i] = 0
            else:
                hour[i] = hour[i-1] + 1
        # days of week
        for i in range(1, len(timeslot)):
            if hour[i-1] == 23:
                dayofweek[i] = dayofweek[i-1] + 1
                if dayofweek[i-1] == 7:
                    dayofweek[i] = 1
            else:
                dayofweek[i] = dayofweek[i-1]
        # days of months, months, years
        for i in range(1, len(timeslot)):
            if hour[i - 1] < 23:
                dayofmonth[i] = dayofmonth[i-1]
                month[i] = month[i-1]
                year[i] = year[i-1]
            else:
                if dayofmonth[i-1] == 28 and month[i-1] == 2:
                        dayofmonth[i] = 1
                        month[i] = 3
                        year[i] = year[i - 1]
                elif dayofmonth[i-1]==30 and (month[i-1]==4 or month[i-1]==6 or month[i-1]==9 or month[i-1]==11):
                        dayofmonth[i] = 1
                        month[i] = month[i-1] + 1
                        year[i] = year[i - 1]
                elif dayofmonth[i-1] == 31:
                    dayofmonth[i] = 1
                    month[i] = month[i - 1] + 1
                    year[i] = year[i - 1]
                    if month[i-1] == 12:
                        month[i] = 1
                        year[i] = year[i-1] + 1
                else:
                    dayofmonth[i] = dayofmonth[i-1]+1
                    month[i] = month[i-1]
                    year[i] = year[i-1]
    if "DistributionUtilityService: ts" in x:
        content = x.split(",")
        temp = content[2]
        temp = temp.replace(' net = ', '')
        temp = temp.replace('\n', '')
        temp = temp.replace('<mwh>', '')
        demand.append(temp)
f.close()

# Transform lists to numpy arrays
timeslot = [float(i) for i in timeslot]
timeslot = np.array(timeslot)
hour = [float(i) for i in hour]
hour = np.array(hour)
dayofweek = [float(i) for i in dayofweek]
dayofweek = np.array(dayofweek)
dayofmonth = [float(i) for i in dayofmonth]
dayofmonth = np.array(dayofmonth)
month = [float(i) for i in month]
month = np.array(month)
year = [float(i) for i in year]
year = np.array(year)
temperature = [float(i) for i in temperature]
temperature = np.array(temperature)
windS = [float(i) for i in windS]
windS = np.array(windS)
windD = [float(i) for i in windD]
windD = np.array(windD)
cloudC = [float(i) for i in cloudC]
cloudC = np.array(cloudC)
temp = demand[0]
temp = temp.replace('  <mwh>', '')
demand[0] = temp
demand = [float(i) for i in demand]
demand = np.array(demand)
for i in range(336):
    demand[i] = demand[i] * 1000

peaks = 0
for i in range(len(demand)):
    temp = []
    for j in range(i):
        temp.append(demand[j])
    threshold.append(calc_threshold(temp))
    if threshold[i] < demand[i]:
        peak.append(1)
        peaks += 1
    else:
        peak.append(0)


data_tuples = list(zip(timeslot, hour, dayofweek, month, year, temperature, windS, windD, cloudC))
df = pd.DataFrame(data_tuples, columns=['tslot', 'hour', 'day', 'month', 'year', 'tmpr', 'windSpd', 'windDir', 'cloudCov'])

# Remove excess datapoints
distance = len(timeslot) - len(demand)
df = df[:-distance]
# # Create lag features
# regr = create_lags(df, lags_num)
regr = df
regr['mwh'] = demand
regr_file_name = './Datasets/finals_2020_11_' + game_num + '.csv'
regr.to_csv(regr_file_name, index=False)

# df['threshold'] = threshold
# clf = create_lags(df, lags_num)
# clf = df
# clf['peak'] = peak
# clf_file_name = './Datasets/finals_2020_11_' + game_num + '_clf.csv'
# clf.to_csv(clf_file_name, index=False)


# if game_num == '1':
#     with open('regression.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Game_Num", "Game_Diff", "Train_Time", "Pred_Time", "Lags", "MSE", "MAE", "RMSE", "R^2"])
#
#     with open('classification.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Game_Num", "Game_Diff", "Train_Time", "Pred_Time", "Lags", "ACC", "PREC", "REC", "F1"])

