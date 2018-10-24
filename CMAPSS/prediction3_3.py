import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# load test data and model
test_df = pd.read_csv('test_data3_3.csv', sep=",", header=0)
model = load_model('lstm_model3_3.h5')

print(test_df)

input_len = 20
# make prediction
com_rul = []
for i in np.arange(1, 101, 1):
    engine = i
    test_X = np.asarray(test_df.loc[test_df['id'] == engine])
    test_X = test_X[:, 1]
    init = test_X
    len_init = len(init)
    #print('init_type', type(init))
    test_X = test_X[-input_len:]
    #print('len_input_sequence', len(test_X))
    while test_X.max() < 0.90:
        test_X_reshaped = test_X.reshape(1, input_len, 1)
        preds = model.predict(test_X_reshaped)
        init = np.concatenate((init, preds[0]))
        #plt.plot(init, '--')
        #plt.plot(init)
        #plt.show()
        test_X = init[-input_len:]
        print('check1')
    plt.plot(init)
    #plt.show()
    new_init = []
    for j in range(len(init)):
        if init[j] < 0.90:
            new_init.append(init[j])
    rul = len(new_init)-len_init
    com_rul.append(rul)
    print('check2 --!!-- ')
#print(init)
plt.show()
print(com_rul)
pd.DataFrame(com_rul).to_csv('RUL3_3.csv', encoding='utf-8', index=None)