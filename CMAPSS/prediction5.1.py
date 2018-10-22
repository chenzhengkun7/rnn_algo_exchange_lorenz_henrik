import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

# load test data and model
test_data = pd.read_csv('test_data5.csv', sep=",", header=0)
rul_data = pd.read_csv('data/rul/RUL_FD001.txt', header=None)
model = load_model('lstm_model5.h5')
scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)

test_data = test_data.drop('index', axis=1)

# chose engine
cycles = []
init_len = []
com_rul = []
for i in np.arange(1, 101, 1):
    engine = i

    # extract test data
    test_extr = test_data.loc[test_data['id'] == engine]
    test_extr.drop(test_extr.columns[[0]], axis=1, inplace=True)
    test_extr = np.asarray(test_extr)

    # reshape input data
    test_X = test_extr.reshape(1, len(test_extr), (test_data.shape[1]-1))

    # Doing the prediction
    preds = model.predict(test_X)
    preds_cyc = scaler.inverse_transform(preds)
    preds_cyc[0].tolist()
    preds_cyc = np.rint(preds_cyc)
    init_len = len(test_extr)

    rul = preds_cyc[0][0] - init_len
    rul = rul.astype(int)
    com_rul.append(rul)
print('com_rul', com_rul)

pd.DataFrame(com_rul).to_csv('RUL5.csv', encoding='utf-8', index=None)


