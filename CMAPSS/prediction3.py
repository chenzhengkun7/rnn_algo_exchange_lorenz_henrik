import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# load test data and model
test_data = pd.read_csv('test_data3.csv', sep=",", header=0)
model = load_model('lstm_model3.h5')
test_data = test_data.drop('index', axis=1)
test_data.columns = ['id', 's2', 's3', 's4', 's11', 's15']

# get and stack single sequences
filt_comp = []
HI = []
max = []
for i in np.arange(1, 101, 1):
    sing_sequ = test_data.loc[test_data['id'] == i]

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

    filtered_mean = filtered_mean.values.tolist()
    HI.extend(filtered_mean)

# generate new engine index and new dataframe with HI
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(test_data.loc[test_data['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)

com_ind = pd.DataFrame(com_ind)
HI = pd.DataFrame(HI)
df = pd.concat([com_ind, HI], axis=1)
df.columns = ['id', 'HI']
print('jackpoint')
input_len = 20
# make prediction
com_rul = []
for i in np.arange(1, 101, 1):
    engine = i
    test_X = np.asarray(df.loc[df['id'] == engine])
    test_X = test_X[:, 1]
    init = test_X
    len_init = len(init)
    #print('init_type', type(init))
    test_X = test_X[-input_len:]
    #print('len_input_sequence', len(test_X))
    while test_X.max() < 0.73:
        test_X_reshaped = test_X.reshape(1, input_len, 1)
        preds = model.predict(test_X_reshaped)
        init = np.concatenate((init, preds[0]))
        #plt.plot(init, '--')
        #plt.plot(init)
        #plt.show()
        test_X = init[-input_len:]
        print('check1')
    #plt.plot(init, '--')
    #plt.show()
    new_init = []
    for j in range(len(init)):
        if init[j] < 0.73:
            new_init.append(init[j])
    rul = len(new_init)-len_init
    com_rul.append(rul)
    print('check2')
print(init)
print(com_rul)
pd.DataFrame(com_rul).to_csv('RUL3.csv', encoding='utf-8', index=None)


