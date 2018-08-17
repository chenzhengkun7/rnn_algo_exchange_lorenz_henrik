import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

#np.random.seed(1234)

names = ['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
         'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
         'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']

# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=names)
df = pd.DataFrame(dataset)
# print(df)
data = pd.DataFrame(columns=names)

for i in range(1, 101):
    # select engine
    engine_no = i
    data_sel = df.loc[df['EngNo'] == engine_no]

    # filter
    window_size = 20
    rolling = data_sel.rolling(window=window_size)
    rm = rolling.mean()
    # delete NaN
    rm = rm.drop(rm.index[0:(window_size - 1)])
    data = pd.concat([data, rm])

data.to_csv('train_FD001_filt.csv', encoding='utf-8', index=None, sep=',')

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
print(train_df)
#plt.plot(train_df['s9'])
#plt.plot(dataset['S9'])
#plt.show()
#####################
# MLP Model
#####################

# define input and output data
X = train_df[['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']]

y = train_df[['class']]

# split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# train the model using the trainings sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

###################
# prepare test data
###################
# select columns of interest
test_df = train_df[['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']]
print(train_df.iloc[12])

########################
# evaluate and use model
########################

# model accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(clf.predict([test_df.iloc[12]]))
