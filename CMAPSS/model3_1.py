import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random

random.seed(196)

# load train data
train_df = pd.read_csv('data/train/train_FD001.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_len = len(train_df)

test_df = pd.read_csv('data/test/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

test_len = len(test_df)

df = pd.concat([train_df, test_df], ignore_index=True)


# get relevant values
id = df['id']
cycle = df['cycle']
s2 = df['s2']
s3 = df['s3']
s4 = df['s4']
s8 = df['s8']
s11 = df['s11']
s13 = df['s13']
s15 = df['s15']
s17 = df['s17']


# create sensor matrix
sensors = pd.concat([cycle, s2, s3, s4, s8, s11, s13, s15], axis=1)
columns = ['cycle', 's2', 's3', 's4', 's8', 's11', 's13', 's15']

# normalize single sensor values
all_sensors = []
for i in range(8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensor = sensors.iloc[:, i].values.reshape(-1, 1)
    scaled = scaler.fit_transform(sensor)
    sensor = pd.DataFrame(scaled).values
    all_sensors.append(sensor)
all_sensors = pd.concat([#pd.DataFrame(all_sensors[0]), #cycle
                         pd.DataFrame(all_sensors[1]), #s2
                         pd.DataFrame(all_sensors[2]), #s3
                         pd.DataFrame(all_sensors[3]), #s4
                         pd.DataFrame(all_sensors[5]), #s11
                         pd.DataFrame(all_sensors[7]) #s15
                         ],
                        axis=1)
#print(all_sensors)

#for i in range(6):
#    plt.plot(all_sensors.iloc[:, i])
#plt.show()

# add id again
comp = pd.concat([id, all_sensors], axis=1)

# split into test and training data again, save test data
train_df = comp.iloc[0:train_len, :]
test_df = comp.iloc[train_len:, :]
test_df = test_df.reset_index()

pd.DataFrame(test_df).to_csv('test_data3_1.csv', encoding='utf-8', index=None)


#extract cycles so that they don't get filtered

# norm  sing cycles
cyc_id = pd.concat([id, cycle], axis=1)
cyc_id = cyc_id.iloc[0:train_len, :]

comp_cyc = []
for i in np.arange(1, 101, 1):
    sing_cyc = cyc_id.loc[cyc_id['id'] == i]
    sing_cyc = sing_cyc.iloc[:, 1].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cyc = sing_cyc
    scaled = scaler.fit_transform(cyc)
    all_cyc = []
    for i in range(len(scaled)):
        cyc = scaled[i]
        all_cyc.extend(cyc)
    comp_cyc.extend(all_cyc)

train_df.columns = columns = ['id', 's2', 's3', 's4', 's11', 's15']
#train_df.drop(train_df.columns[[1]], axis=1, inplace=True)

# get and stack single sequences
filt_comp = []
HI = []
max = []
for i in np.arange(1, 101, 1):
    sing_sequ = train_df.loc[train_df['id'] == i]

    # filter data
    filtered = []
    for i in np.arange(1, 6, 1):
        sing_sequ_sens = sing_sequ.iloc[:, i].values
        window = 10
        rolling = pd.Series(sing_sequ_sens).rolling(window=window)
        rolling_mean = rolling.mean()
        rolling_mean = rolling_mean[window:]
        rolling_mean = rolling_mean.tolist()
        filtered.append(rolling_mean)

    filtered = np.asarray(filtered)
    filtered = pd.DataFrame(filtered)
    filtered_mean = filtered.mean(axis=0)
    max.append(filtered_mean.max())
    #print(filtered)

    #for i in np.arange(0, 5, 1):
    #   plt.plot(filtered.iloc[i, :], 'k')
    #plt.show()


    #for i in np.arange(0, 5, 1):
    #    plt.plot(filtered_mean, 'r')
    #plt.show()

    filtered_mean = filtered_mean.values.tolist()
    HI.extend(filtered_mean)
#plt.plot(HI)
#plt.show()
maxes = (np.asarray(max))

# threshold for RUL
max = np.mean(maxes)
print('max', max)

# cut cycles due to filtering
cyc_id = id
comp_cyc = pd.DataFrame(comp_cyc)
cyc_id = cyc_id.iloc[0:(len(train_df)), ]
norm_cyc = comp_cyc.iloc[0:(len(train_df)), ]
cycles = pd.concat([cyc_id, norm_cyc], axis=1)

com_cyc = []
for i in np.arange(1, 101, 1):
    sing_cyc = cycles.loc[cycles['id'] == i]
    new_cyc = sing_cyc.iloc[window:, :]
    new_cyc = new_cyc.iloc[:, 1].tolist()
    com_cyc.extend(new_cyc)

#plt.plot(com_cyc)
#plt.show()

# generate new engine index and new dataframe with HI
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(train_df.loc[train_df['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)


com_ind = pd.DataFrame(com_ind)
HI = pd.DataFrame(HI)
com_cyc = pd.DataFrame(com_cyc)
df = pd.concat([com_ind, com_cyc, HI], axis=1)
df.columns = ['id', 'cycles', 'HI']

#print(df.loc[df['id'] == 85])
#plt.plot(df.loc[df['id'] == 1])
#plt.show()

# create sequences
comp_sequ = []
for i in np.arange(1, 101, 1):
    new_sequ = []
    n = 1
    sequ = df.loc[df['id'] == i]
    sequ.drop(sequ.columns[[0]], axis=1, inplace=True)
    seq_len = len(sequ)
    sing_sequ = sequ[-25:(seq_len)]
    while len(sing_sequ) == 25:
        new_sequ.append(sing_sequ)
        sing_sequ = sequ[-25-(n*20):(seq_len-(n*20))]
        n = n + 1
    comp_sequ.extend(new_sequ)

print(len(comp_sequ))
plt.plot(comp_sequ[0])
plt.show()

# get input
input = np.empty((0, 20, 2))
for i in range(len(comp_sequ)):
    sing_sequ = comp_sequ[i]
    sing_sequ = np.asarray(sing_sequ)
    X = sing_sequ[0:20, :]
    input = np.append(input, [X], axis=0)

# get output (HI)
output = []
for i in range(len(comp_sequ)):
    sing_sequ = comp_sequ[i]
    sing_sequ = np.asarray(sing_sequ)
    y_com = sing_sequ[20:25, :]
    Y = y_com[:, 1]
    output.append(Y)
output = np.asarray(output)

# build lstm-model
model = Sequential()
model.add(LSTM(150, input_shape=(None, 2)))
model.add(Dense(5))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(input, output, epochs=20, batch_size=1, verbose=2)
model.save('lstm_model3_1.h5')




