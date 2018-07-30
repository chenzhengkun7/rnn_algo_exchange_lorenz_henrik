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
#print(len(df))

# select engine
engine_no = 5
data_sel = df.loc[df['EngNo'] == engine_no]
#print(len(data_sel))

# select exponential features 2, 3, 4, 8, 11, 13, 15, 17
feature_sel2 = data_sel.loc[:, 'S2']
feature_sel3 = data_sel.loc[:, 'S3']
feature_sel4 = data_sel.loc[:, 'S4']
feature_sel8 = data_sel.loc[:, 'S8']
feature_sel11 = data_sel.loc[:, 'S11']
feature_sel13 = data_sel.loc[:, 'S13']
feature_sel15 = data_sel.loc[:, 'S15']
feature_sel17 = data_sel.loc[:, 'S17']

y = np.column_stack((feature_sel2, feature_sel3, feature_sel4, feature_sel8, feature_sel11,
                    feature_sel13, feature_sel15, feature_sel17))

y_new = y.mean(1)

# normalisation
series = Series(y_new)

values = series.values
values = values.reshape((len(values), 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)

normalized = scaler.transform(values)

# linear regression
x = np.linspace(0, len(data_sel), num=len(data_sel))[:, None]

x_new = np.hstack([x, x**2, x**3, x**4])

model = LinearRegression()
model.fit(x_new, normalized)

y_pred = model.predict(x_new)

plt.plot(normalized, 'k+')
#plt.plot(normalized2, 'k+')
#plt.plot(normalized3, 'k+')
#plt.plot(normalized4, 'k+')
#plt.plot(normalized8, 'k+')
#plt.plot(normalized11, 'k+')
#plt.plot(normalized13, 'k+')
#plt.plot(normalized15, 'k+')
#plt.plot(normalized17, 'k+')
plt.plot(y_pred, 'r')
plt.show()
