import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# load test data and model
test_data = pd.read_csv('test_data3_1.csv', sep=",", header=0)
model = load_model('lstm_model3_1.h5')
test_data = test_data.drop('index', axis=1)
test_data.columns = ['id', 's2', 's3', 's4', 's11', 's15']

id = test_data['id']

#extract cycles so that they don't get filtere
test_df = test_data
norm_cyc = test_df.iloc[:, 1]
test_df.columns = columns = ['id', 's2', 's3', 's4', 's11', 's15']
#test_df.drop(test_df.columns[[1]], axis=1, inplace=True)

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

# cut cycles due to filtering
cyc_id = id
cyc_id = cyc_id.iloc[0:(len(test_df)), ]
norm_cyc = norm_cyc.iloc[0:(len(test_df)), ]
cycles = pd.concat([cyc_id, norm_cyc], axis=1)

com_cyc = []
for i in np.arange(1, 101, 1):
    sing_cyc = cycles.loc[cycles['id'] == i]
    new_cyc = sing_cyc.iloc[window:, :]
    new_cyc = new_cyc.iloc[:, 1].tolist()
    com_cyc.extend(new_cyc)

# generate new engine index and new dataframe with HI
com_ind = []
for i in np.arange(1, 101, 1):
    len_id = len(test_data.loc[test_data['id'] == i]) - window
    new_index = np.full(len_id, i).tolist()
    com_ind.extend(new_index)

com_ind = pd.DataFrame(com_ind)
HI = pd.DataFrame(HI)
com_cyc = pd.DataFrame(com_cyc)

df = pd.concat([com_ind, com_cyc, HI], axis=1)
df.columns = ['id', 'cycles', 'HI']

# create sequences
comp_sequ = []
input = np.empty((0, 20, 2))
#for i in np.arange(1, 101, 1):
n = 1
for i in np.arange(1, 101, 1):
    sequ = df.loc[df['id'] == i]
    sequ.drop(sequ.columns[[0]], axis=1, inplace=True)
    seq_len = len(sequ)
    sing_sequ = sequ[-20:(seq_len)]
    sing_sequ = np.asarray(sing_sequ)

    # create 3D input array
    input = np.append(input, [sing_sequ], axis=0)

print(input.shape)


# make prediction
com_rul = []
for i in np.arange(1, 100, 1):
    engine = i
    test_X = input[engine]
    init = input[engine]
    #init = init[:, 1]


    # get straight equation
    x_str = np.arange(0, 20, 1)
    y_str = test_X[:, 0]
    coefficients = np.polyfit(x_str, y_str, 1)
    n = 1
    while test_X.max() < 0.73:
        test_X = test_X.reshape(1, (15 + (n * 5)), 2)
        preds = model.predict(test_X)
        poly = np.poly1d(coefficients)
        x_axis = np.arange(15 + (n*5), 20 + (n*5), 1)
        y_axis = poly(x_axis)
        preds = np.vstack((y_axis, preds[0]))
        preds = preds.T

        plt.plot(preds)
        plt.show()

        init = np.vstack((init, preds))
        test_X = init
        n = n + 1
    #print(init[:, 1])
    #print(len(init[:, 1]))
    com_rul.append(len(init[:, 1]))

print(com_rul)
print(len(com_rul))
pd.DataFrame(com_rul).to_csv('RUL3_1.csv', encoding='utf-8', index=None)


