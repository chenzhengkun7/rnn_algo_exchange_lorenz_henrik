import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random


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
sensors = pd.concat([s2, s3, s4, s8, s11, s13, s15], axis=1)
columns = ['s2', 's3', 's4', 's8', 's11', 's13', 's15']

# normalize single sensor values
all_sensors = []
for i in range(7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensor = sensors.iloc[:, i].values.reshape(-1, 1)
    scaled = scaler.fit_transform(sensor)
    sensor = pd.DataFrame(scaled).values
    all_sensors.append(sensor)
all_sensors = pd.concat([pd.DataFrame(all_sensors[0]),
                         pd.DataFrame(all_sensors[1]),
                         pd.DataFrame(all_sensors[2]),
                         pd.DataFrame(all_sensors[4]),
                         pd.DataFrame(all_sensors[6])
                         ],
                        axis=1)
#print(all_sensors)

#for i in range(5):
#    plt.plot(all_sensors.iloc[:, i])
#plt.show()

comp = pd.concat([id, all_sensors], axis=1)

# split into test and training data again, save test data
train_df = comp.iloc[0:train_len, :]
test_df = comp.iloc[train_len:, :]
test_df = test_df.reset_index()

pd.DataFrame(test_df).to_csv('test_data3.csv', encoding='utf-8', index=None)
test_df = test_df.drop('index', axis=1)
print(test_df.head(32))
plt.plot(test_df)
plt.show()

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

# generate new engine index and new dataframe with HI
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(train_df.loc[train_df['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)

com_ind = pd.DataFrame(com_ind)
HI = pd.DataFrame(HI)
df = pd.concat([com_ind, HI], axis=1)
df.columns = ['id', 'HI']

# create sequences
sequences = []
for i in np.arange(1, 101, 1):
    comp_sequ = []
    new_sequ = []
    n = 1
    sequ = df['HI'].loc[df['id'] == i]
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
len_train = int(tot_len * 0.9)
train_X = input #[0:len_train]
train_Y = output #[0:len_train]
#test_X = input[len_train:]
#test_Y = output[len_train:]
#print(train_X.shape)
#print(train_Y.shape)
#print(test_X.shape)
#print(test_Y.shape)
#for i in range(train_X.shape[0]):
#    plt.plot(train_X[i])
#plt.show()


# reshape input data
train_X_reshaped = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
#test_X_reshaped = test_X.reshape(test_X.shape[0], 15, 1)

# build lstm-model
model = Sequential()
model.add(LSTM(150, input_shape=(None, 1)))
model.add(Dense(5))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X_reshaped, train_Y, epochs=20, batch_size=1, verbose=2)
model.save('lstm_model3.h5')

# Doing a prediction
train_X = train_X[1]
#test_X = test_X[1]
#init = test_X
#test_Y = test_Y[1]
#for i in range(10):
#    test_X_reshaped = test_X.reshape(1, len(test_X), 1)
#    preds = model.predict(test_X_reshaped)
#    x_preds = np.arange(len(test_X), (len(test_X)+10), 1)
#    test_X = np.concatenate([test_X, preds[0]])
#plt.plot(test_X)
#raw = np.concatenate([init, test_Y])
#plt.plot(raw)
#plt.show()


