import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression

# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
                                                         'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
                                                         'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21'])
df = pd.DataFrame(dataset)
#print(df)

# select engine
engine_no = 1
data_sel = df.loc[df['EngNo'] == engine_no]
# print(data_sel)

# select exponential features 2, 3, 4, 8, 11, 13, 15, 17
feature_sel2 = data_sel.loc[:, 'S2']
feature_sel3 = data_sel.loc[:, 'S3']
feature_sel4 = data_sel.loc[:, 'S4']
feature_sel8 = data_sel.loc[:, 'S8']
feature_sel11 = data_sel.loc[:, 'S11']
feature_sel13 = data_sel.loc[:, 'S13']
feature_sel15 = data_sel.loc[:, 'S15']
feature_sel17 = data_sel.loc[:, 'S17']
x = data_sel.loc[:, 'Cycle']

# Normalization
series2 = Series(feature_sel2)
series3 = Series(feature_sel3)
series4 = Series(feature_sel4)
series8 = Series(feature_sel8)
series11 = Series(feature_sel11)
series13 = Series(feature_sel13)
series15 = Series(feature_sel15)
series17 = Series(feature_sel17)

# prepare data for normalization
values2 = series2.values
values2 = values2.reshape((len(values2), 1))
values3 = series3.values
values3 = values3.reshape((len(values3), 1))
values4 = series4.values
values4 = values4.reshape((len(values4), 1))
values8 = series8.values
values8 = values8.reshape((len(values8), 1))
values11 = series11.values
values11 = values11.reshape((len(values11), 1))
values13 = series13.values
values13 = values13.reshape((len(values13), 1))
values15 = series15.values
values15 = values15.reshape((len(values15), 1))
values17 = series17.values
values17 = values17.reshape((len(values17), 1))

# train the normalization
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2 = scaler2.fit(values2)
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaler3 = scaler3.fit(values3)
scaler4 = MinMaxScaler(feature_range=(0, 1))
scaler4 = scaler4.fit(values4)
scaler8 = MinMaxScaler(feature_range=(0, 1))
scaler8 = scaler8.fit(values8)
scaler11 = MinMaxScaler(feature_range=(0, 1))
scaler11 = scaler11.fit(values11)
scaler13 = MinMaxScaler(feature_range=(0, 1))
scaler13 = scaler13.fit(values13)
scaler15 = MinMaxScaler(feature_range=(0, 1))
scaler15 = scaler15.fit(values15)
scaler17 = MinMaxScaler(feature_range=(0, 1))
scaler17 = scaler17.fit(values17)
# print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

# normalize the dataset and print
normalized2 = scaler2.transform(values2)
normalized3 = scaler3.transform(values3)
normalized4 = scaler4.transform(values4)
normalized8 = scaler8.transform(values8)
normalized11 = scaler11.transform(values11)
normalized13 = scaler13.transform(values13)
normalized15 = scaler15.transform(values15)
normalized17 = scaler17.transform(values17)


x = data_sel.loc[:, 'Cycle']

y = np.column_stack((normalized2, normalized3, normalized4, normalized8, normalized11,
                    normalized13, normalized15, normalized17))

#plt.plot(normalized2, 'k+')
#plt.plot(normalized3, 'k+')
#plt.plot(normalized4, 'k+')
#plt.plot(normalized8, 'k+')
#plt.plot(normalized11, 'k+')
#plt.plot(normalized13, 'k+')
#plt.plot(normalized15, 'k+')
#plt.plot(normalized17, 'k+')

#plt.show()

# mean calculation
y_new = y.mean(1)

# linear regression
x = np.linspace(0, len(data_sel), num=len(data_sel))[:, None]

x_new = np.hstack([x, x**2, x**3, x**4])

model = LinearRegression()
model.fit(x_new, y_new)

y_pred = model.predict(x_new)

plt.plot(y_new, 'k+')
plt.plot(y_pred, 'r')
plt.show()

