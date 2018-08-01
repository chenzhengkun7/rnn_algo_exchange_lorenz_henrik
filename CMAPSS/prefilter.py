import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
from scipy import signal

# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
                                                         'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
                                                         'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21'])
df = pd.DataFrame(dataset)
#print(df)

for i in range(1, 100):
    # select engine
    engine_no = i
    data_sel = df.loc[df['EngNo'] == engine_no]

    # select essential features
    feature_sel2 = data_sel.loc[:, 'S2']
    feature_sel3 = data_sel.loc[:, 'S3']
    feature_sel4 = data_sel.loc[:, 'S4']
    feature_sel8 = data_sel.loc[:, 'S8']
    feature_sel11 = data_sel.loc[:, 'S11']
    feature_sel13 = data_sel.loc[:, 'S13']
    feature_sel15 = data_sel.loc[:, 'S15']
    feature_sel17 = data_sel.loc[:, 'S17']
    x = data_sel.loc[:, 'Cycle']

    # butterworth filter
    b, a = signal.butter(3, 0.05)
    ff2 = signal.filtfilt(b, a, feature_sel2)
    ff3 = signal.filtfilt(b, a, feature_sel3)
    ff4 = signal.filtfilt(b, a, feature_sel4)
    ff8 = signal.filtfilt(b, a, feature_sel8)
    ff11 = signal.filtfilt(b, a, feature_sel11)
    ff13 = signal.filtfilt(b, a, feature_sel13)
    ff15 = signal.filtfilt(b, a, feature_sel15)
    ff17 = signal.filtfilt(b, a, feature_sel17)

    # Normalization
    series2 = Series(ff2)
    series3 = Series(ff3)
    series4 = Series(ff4)
    series8 = Series(ff8)
    series11 = Series(ff11)
    series13 = Series(ff13)
    series15 = Series(ff15)
    series17 = Series(ff17)

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

    # normalize the dataset
    normalized2 = scaler2.transform(values2)
    normalized3 = scaler3.transform(values3)
    normalized4 = scaler4.transform(values4)
    normalized8 = scaler8.transform(values8)
    normalized11 = scaler11.transform(values11)
    normalized13 = scaler13.transform(values13)
    normalized15 = scaler15.transform(values15)
    normalized17 = scaler17.transform(values17)

    y = np.column_stack((normalized2, normalized3, normalized4, normalized8, normalized11,
                            normalized13, normalized15, normalized17))

    #plt.plot(y)
    #plt.show()

    y_new = y.mean(1)

    # linear regression
    x = np.linspace(0, len(data_sel), num=len(data_sel))[:, None]

    x_new = np.hstack([x, x ** 2, x ** 3, x ** 4])

    model = LinearRegression()
    model.fit(x_new, y_new)

    y_pred = model.predict(x_new)

    plt.plot(y_new, 'k+')
    plt.plot(y_pred, 'r')

plt.show()


