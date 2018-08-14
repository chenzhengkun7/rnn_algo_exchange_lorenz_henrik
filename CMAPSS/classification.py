import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(1234)

# read training data - It is the aircraft engine run-to-failure data.
train_df = pd.read_csv('train_FD001_filt.csv', sep=" ", header=0)
#train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
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
# Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
# RUL der Trainingsdaten wird ausgelesen
train_df = train_df.merge(rul, on=['id'], how='left')
# max RUL wird den Enginges zugeordnet
train_df['RUL'] = train_df['max'] - train_df['cycle']
# RUL wird berechnet und der Train Matrix angefügt
train_df.drop('max', axis=1, inplace=True)

# generate label columns for training data
for i in range(100):
    sel_eng = train_df[(train_df.id == 1)]
    no_eng = len(sel_eng)
    classes = np.linspace(0, no_eng, 11)
    cl_round = np.round(classes)
    for j in range(10):
        train_df.loc[(train_df['cycle'] > cl_round[j]) & (train_df['id'] == i + 1), 'class'] = j + 1

# sort new dataframe regarding class
train_df = train_df.sort_values(by=['class'])
print(train_df)

# extract classes
class_1 = train_df.loc[train_df['class'] == 1]
class_2 = train_df.loc[train_df['class'] == 2]
class_3 = train_df.loc[train_df['class'] == 3]
class_4 = train_df.loc[train_df['class'] == 4]
class_5 = train_df.loc[train_df['class'] == 5]
class_6 = train_df.loc[train_df['class'] == 6]
class_7 = train_df.loc[train_df['class'] == 7]
class_8 = train_df.loc[train_df['class'] == 8]
class_9 = train_df.loc[train_df['class'] == 9]
class_10 = train_df.loc[train_df['class'] == 10]

size_list = [class_1.shape, class_2.shape, class_3.shape, class_4.shape, class_5.shape, class_6.shape,
             class_7.shape, class_8.shape, class_9.shape, class_10.shape]

min_value = min(size_list)
min_value = min_value[0]

# resample data
class_1 = class_1.sample(n=min_value)
class_2 = class_2.sample(n=min_value)
class_3 = class_3.sample(n=min_value)
class_4 = class_4.sample(n=min_value)
class_5 = class_5.sample(n=min_value)
class_6 = class_6.sample(n=min_value)
class_7 = class_7.sample(n=min_value)
class_8 = class_8.sample(n=min_value)
class_9 = class_9.sample(n=min_value)
class_10 = class_10.sample(n=min_value)

data = class_1.append(class_2, ignore_index=True)
data = data.append(class_3, ignore_index=True)
data = data.append(class_4, ignore_index=True)
data = data.append(class_5, ignore_index=True)
data = data.append(class_6, ignore_index=True)
data = data.append(class_7, ignore_index=True)
data = data.append(class_8, ignore_index=True)
data = data.append(class_9, ignore_index=True)
data = data.append(class_10, ignore_index=True)
data.to_csv('labeled_data.csv', encoding='utf-8', index=None)

#####################
# Random Forest Model
#####################

# define input and output data
X = data[['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']]

y = data[['class']]

# split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# train the model using the trainings sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

###################
# prepare test data
###################
# select columns of interest
#test_df = test_df[['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
#                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
#                   's15', 's16', 's17', 's18', 's19', 's20', 's21']]

########################
# evaluate and use model
########################

# model accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#print(clf.predict([test_df.iloc[125]]))
