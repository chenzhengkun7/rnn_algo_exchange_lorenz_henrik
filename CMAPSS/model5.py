import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.externals import joblib

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
in_data = df.iloc[:, 1:]
#plt.plot(in_data.iloc[:, 24])
#plt.show()
id = df.iloc[:, 0]

# normalize single input values
all_input = []
for i in np.arange(1, 25, 1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensor = in_data.iloc[:, i].values.reshape(-1, 1)
    scaled = scaler.fit_transform(sensor)
    sensor = pd.DataFrame(scaled).values
    all_input.append(sensor)

# normalize cycles aka outputs
scaler = MinMaxScaler(feature_range=(0, 1))
sensor = in_data.iloc[:, 0].values.reshape(-1, 1)
scaled = scaler.fit_transform(sensor).tolist()

# save scaler for use in prediction model
scaler_filename = 'scaler.save'
joblib.dump(scaler, scaler_filename)

all_input = pd.concat([pd.DataFrame(scaled), #cycle
                       pd.DataFrame(all_input[0]), #setting1
                       pd.DataFrame(all_input[1]), #setting2
                       #pd.DataFrame(all_input[2]), #setting2
                       #pd.DataFrame(all_input[3]), #s1
                       pd.DataFrame(all_input[4]), #s2
                       pd.DataFrame(all_input[5]), #s3
                       pd.DataFrame(all_input[6]), #s4
                       #pd.DataFrame(all_input[7]), #s5
                       #pd.DataFrame(all_input[8]), #s6
                       pd.DataFrame(all_input[0]),#s7
                       pd.DataFrame(all_input[10]),#s8
                       pd.DataFrame(all_input[11]),#s9
                       #pd.DataFrame(all_input[12]),#s10
                       pd.DataFrame(all_input[13]),#s11
                       pd.DataFrame(all_input[14]),#s12
                       pd.DataFrame(all_input[15]),#s13
                       pd.DataFrame(all_input[16]),#s14
                       pd.DataFrame(all_input[17]),#s15
                       #pd.DataFrame(all_input[18]),#s16
                       pd.DataFrame(all_input[19]),#s17
                       #pd.DataFrame(all_input[20]),#s18
                       #pd.DataFrame(all_input[21]),#s19
                       pd.DataFrame(all_input[22]),#s20
                       pd.DataFrame(all_input[23]),#s21
                       ], axis=1)

input_data = pd.concat([id, all_input], axis=1)
# split into test and training data again, save test data
train_df = input_data.iloc[0:train_len, :]
test_df = input_data.iloc[train_len:, :]
test_df = test_df.reset_index()

pd.DataFrame(test_df).to_csv('test_data5.csv', encoding='utf-8', index=None)

# get min max length of cycles
len_list = []
for i in np.arange(1, 101, 1):
    len_list = np.asarray(len_list)
    train_sel = train_df.loc[train_df['id'] == i]
    train_sel = train_sel.iloc[:, 1]
    length = len(train_sel)
    length = np.asarray(length)
    len_list = np.append(len_list, length)
len_list = np.asarray(len_list)
min = np.min(len_list)
max = np.max(len_list)

#train_df = np.asarray(train_df)

# get and stack single sequences
sequences = []
for i in np.arange(1, 101, 1):
    sing_sequ = train_df.loc[train_df['id'] == i]
    sequences.append(sing_sequ)

new_sequences = []
for i in range(100):
    new_sequences.append(sequences[i][0])
#for i in range(100):
#    plt.plot(new_sequences[i])
#plt.show()

#define window for filter
window = 10
# extract cycles because they should not be filtered but need to be cut to window size
com_cyc = []
for i in range(100):
    cycles = np.asarray(new_sequences[i])
    cycles = cycles[:, 0].tolist()
    com_cyc.append(cycles)

# create output data y
y = []
for i in range(100):
    cyc = com_cyc[i]
    sing_y = cyc[-1]
    y.append(sing_y)
y = np.asarray(y)

sing_y = y[-5]
sing_y = sing_y.reshape(1, -1)
inv_y = scaler.inverse_transform(sing_y)

# cut of cycles from new_sequences for filter preparation
fil_data = []
for i in range(100):
    sing = np.asarray(new_sequences[i])
    sing = np.delete(sing, 0, 1).tolist()
    fil_data.append(sing)
fil_data = np.asarray(fil_data)

# filter
filtered = []
for i in range(100):
    rolling = pd.DataFrame(fil_data[i]).rolling(window=window)
    rolling_mean = rolling.mean()
    rolling_mean = rolling_mean.dropna()
    filtered.append(rolling_mean)

# add cycles again
seq = []
for i in range(100):
    sing_cyc = pd.DataFrame(com_cyc[i])
    sing_fil = filtered[i]
    com_seq = pd.concat([sing_cyc, sing_fil], axis=1)
    com_seq = com_seq.iloc[9:, :]
    seq.append(com_seq)

#print(seq)
#for i in range(100):
#    plt.plot(seq[i])
#plt.show()

# get input sequences
input_sequ = []
for i in range(100):
    cut_sequ = seq[i][0:119]
    input_sequ.append(cut_sequ)

#print(len(input_sequ[0]))
#print(input_sequ)
#for i in range(100):
#    plt.plot(input_sequ[i])
#plt.show()

input_array = np.empty((0, 119, all_input.shape[1]))
for i in range(100):
    single_sample = np.asarray(input_sequ[i])
    input_array = np.append(input_array, [single_sample], axis=0)

#plt.plot(input_array[0])
#plt.show()
#print(input_array.shape)
#print(y[0])

# split into train and test data/ in- and outputs
train_X = input_array
train_y = y

# create model
model = Sequential()
model.add(LSTM(150, input_shape=(None, all_input.shape[1])))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_X, train_y, validation_split=0.01, epochs=20, batch_size=1, verbose=2)
model.save('lstm_model5.h5')

# Doing the prediction
#preds = model.predict(test_X)
#inv_preds = y_scaler.inverse_transform(preds)
#inv_test_y = y_scaler.inverse_transform(test_y)
#print(preds)
#print(test_y)

# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

