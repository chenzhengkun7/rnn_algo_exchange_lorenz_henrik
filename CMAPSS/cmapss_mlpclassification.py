from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

seed = 7
np.random.seed(seed)

raw_df = pd.read_csv('data/train/train_FD001.txt', sep=" ", header=None)
raw_df.drop(raw_df.columns[[26, 27]], axis=1, inplace=True)
raw_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read training data - It is the aircraft engine run-to-failure data.
train_df = pd.read_csv('train_FD001_filt.csv', sep=",", header=0)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']


# read test data - It is the aircraft engine operating data without failure events recorded.
test_df = pd.read_csv('data/test/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

##################################
# Data Preprocessing
##################################

#######
# TRAIN
#######

# generate label columns for training data
for i in range(100):
    sel_eng = train_df[(train_df.id == 1)]
    no_eng = len(sel_eng)
    classes = np.linspace(0, no_eng, 11)
    cl_round = np.round(classes)
    for j in range(10):
        train_df.loc[(train_df['cycle'] > cl_round[j]) & (train_df['id'] == i + 1), 'class'] = j + 1

#print(raw_df.shape)
#print(train_df.shape)
#plt.plot(raw_df['s9'])
#plt.plot(train_df['s9'])
#plt.show()

# define input and output data
X = train_df[['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21']]

y = train_df[['class']]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(48, input_dim=24, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

# evaluation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))