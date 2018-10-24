import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random

random.seed(3)

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
s2 = df['s2']
s3 = df['s3']
s4 = df['s4']
s8 = df['s8']
s11 = df['s11']
s13 = df['s13']
s15 = df['s15']
s17 = df['s17']

# create sensor matrix
sensors = pd.concat([id, s2, s4, s11, s15], axis=1)
columns = ['id', 's2', 's4', 's11', 's15']

# split into test and training data again
train_df = sensors.iloc[0:train_len, :]
test_df = sensors.iloc[train_len:, :]
test_df = test_df.reset_index(drop=True)

# filter train data
filt_comp = []
HI = []
for i in np.arange(1, 101, 1):
    sing_sequ = train_df.loc[train_df['id'] == i]

    # filter data
    filtered = []
    for i in np.arange(1, 5, 1):
        sing_sequ_sens = sing_sequ.iloc[:, i].values
        window = 10
        rolling = pd.Series(sing_sequ_sens).rolling(window=window)
        rolling_mean = rolling.mean()
        rolling_mean = rolling_mean[window:]
        rolling_mean = rolling_mean.tolist()
        filtered.append(rolling_mean)

    filtered = np.asarray(filtered)
    filtered = filtered.T
    filt_comp.extend(filtered)
filt_comp = np.asarray(filt_comp)
filt_train = pd.DataFrame(filt_comp)

# filter test data
filt_comp = []
HI = []
for i in np.arange(1, 101, 1):
    sing_sequ = test_df.loc[test_df['id'] == i]
    # filter data
    filtered = []
    for i in np.arange(1, 5, 1):
        sing_sequ_sens = sing_sequ.iloc[:, i].values
        window = 10
        rolling = pd.Series(sing_sequ_sens).rolling(window=window)
        rolling_mean = rolling.mean()
        rolling_mean = rolling_mean[window:]
        rolling_mean = rolling_mean.tolist()
        filtered.append(rolling_mean)

    filtered = np.asarray(filtered)
    filtered = filtered.T
    filt_comp.extend(filtered)
filt_comp = np.asarray(filt_comp)
filt_test = pd.DataFrame(filt_comp)

filtered = pd.concat([filt_train, filt_test], axis=0)

# normalize single sensor values
all_sensors = []
for i in range(4):
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensor = filtered.iloc[:, i].values.reshape(-1, 1)
    scaled = scaler.fit_transform(sensor)
    sensor = pd.DataFrame(scaled).values
    all_sensors.append(sensor)
all_sensors = pd.concat([pd.DataFrame(all_sensors[0]),
                         pd.DataFrame(all_sensors[1]),
                         pd.DataFrame(all_sensors[2]),
                         pd.DataFrame(all_sensors[3]),
                         ],
                        axis=1)

#print(all_sensors)
#for i in range(4):
#    plt.plot(all_sensors.iloc[:, i])
#plt.show()

HI = all_sensors.mean(axis=1)

HI_train = HI.loc[:(train_len-(100*window)-1)]
HI_test = HI.loc[(train_len-(100*window)):]
HI_test.reset_index(drop=True, inplace=True)
#plt.plot(HI_train)
#plt.plot(HI_test)
#plt.show()

# generate new engine index and new dataframe with HI_train
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(train_df.loc[train_df['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)

train_id = pd.DataFrame(com_ind)
train_df = pd.concat([train_id, HI_train], axis=1)
train_df.columns = ['id', 'HI']

# generate new engine index and new dataframe with HI_test
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(test_df.loc[test_df['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)

test_id = pd.DataFrame(com_ind)
test_df = pd.concat([test_id, HI_test], axis=1)
test_df.columns = ['id', 'HI']

pd.DataFrame(test_df).to_csv('test_data3_3.csv', encoding='utf-8', index=None)

# threshold for RUL
all_max = []
for i in np.arange(1, 101, 1):
    sing_max = train_df['HI'].loc[train_df['id'] == i].max()
    all_max.append(sing_max)
all_max = np.asarray(all_max)
max = np.mean(all_max)
print('max', max)

# create sequences
sequences = []
for i in np.arange(1, 101, 1):
    comp_sequ = []
    new_sequ = []
    n = 1
    sequ = train_df['HI'].loc[train_df['id'] == i]
    seq_len = len(sequ)
    sing_sequ = sequ[-25:(seq_len)]
    while len(sing_sequ) == 25:
        comp_sequ.append(sing_sequ)
        sing_sequ = sequ[-25-(n*20):(seq_len-(n*20))]
        n = n + 1
    sequences.extend(comp_sequ)
sequences = np.asarray(sequences)
#np.random.shuffle(sequences)
#print(len(sequences))

#for i in np.arange(1, len(sequences), 1):
#    plt.plot(sequences[i])
#plt.show()

tot_len = sequences.shape[0]
plt.plot(sequences[1], 'r')

# split into input and output
input = []
for i in range(tot_len):
    sing_in = sequences[i][0:20].tolist()
    input.append(sing_in)

input = np.asarray(input)
#print('in shape', input.shape)
#print(len(input))
#for i in range(100):
#    plt.plot(input[i])
#plt.show()

plt.plot(input[1])

output = []
for i in range(tot_len):
    sing_out = sequences[i][-5:25].tolist()
    output.append(sing_out)

output = np.asarray(output)
#print('out shape', output.shape)
#for i in range(100):
#    plt.plot(output[i])
#plt.show()


# split test train
len_train = int(tot_len)
train_X = input
train_Y = output


# reshape input data
train_X_reshaped = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)

# build lstm-model
model = Sequential()
model.add(LSTM(150, input_shape=(None, 1)))
model.add(Dense(5))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X_reshaped, train_Y, epochs=20, batch_size=1, verbose=2)
model.save('lstm_model3_3.h5')




