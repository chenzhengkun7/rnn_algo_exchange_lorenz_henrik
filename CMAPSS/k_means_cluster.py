import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans

# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
                                                         'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
                                                         'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21'])
df = pd.DataFrame(dataset)
#print(df)

for i in range(1,100):
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

    # filter
    window_size = 20
    rolling2 = feature_sel2.rolling(window=window_size)
    rm2 = rolling2.mean()
    rolling3 = feature_sel3.rolling(window=window_size)
    rm3 = rolling3.mean()
    rolling4 = feature_sel4.rolling(window=window_size)
    rm4 = rolling4.mean()
    rolling8 = feature_sel8.rolling(window=window_size)
    rm8 = rolling8.mean()
    rolling11 = feature_sel11.rolling(window=window_size)
    rm11 = rolling11.mean()
    rolling13 = feature_sel13.rolling(window=window_size)
    rm13 = rolling13.mean()
    rolling15 = feature_sel15.rolling(window=window_size)
    rm15 = rolling15.mean()
    rolling17 = feature_sel17.rolling(window=window_size)
    rm17 = rolling17.mean()

    # delete NaN
    rm2 = rm2.drop(rm2.index[0:(window_size-1)])
    rm3 = rm3.drop(rm3.index[0:(window_size - 1)])
    rm4 = rm4.drop(rm4.index[0:(window_size - 1)])
    rm8 = rm8.drop(rm8.index[0:(window_size - 1)])
    rm11 = rm11.drop(rm11.index[0:(window_size - 1)])
    rm13 = rm13.drop(rm13.index[0:(window_size - 1)])
    rm15 = rm15.drop(rm15.index[0:(window_size - 1)])
    rm17 = rm17.drop(rm17.index[0:(window_size - 1)])


    # Normalization
    series2 = Series(rm2)
    series3 = Series(rm3)
    series4 = Series(rm4)
    series8 = Series(rm8)
    series11 = Series(rm11)
    series13 = Series(rm13)
    series15 = Series(rm15)
    series17 = Series(rm17)

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

    y = np.column_stack((normalized2, normalized3, normalized4, normalized8, normalized11,
                         normalized13, normalized15, normalized17))

    y_new = np.asarray(pd.DataFrame(y.mean(1)))
    #print(y_new)
    x_new = np.arange(0, len(y_new), 1)
    #print(x_new)
    y = np.column_stack((x_new, y_new))
    #print(y)



    # k-means cluster
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(y)

    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_)

    plt.scatter(y[:, 0], y[:, 1], c=kmeans.labels_, cmap='rainbow')
    #plt.scatter(x_new, y_new, c=kmeans.labels_, cmap='rainbow')
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.show()