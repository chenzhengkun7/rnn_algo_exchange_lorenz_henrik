# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:45:26 2018

@author: henri

Funktioniert noch nicht, Evtl Klassen Labeln??
"""

import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
                                                         'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
                                                         'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21'])
#df = pd.DataFrame(dataset)

df = dataset.loc[dataset['EngNo'] == 1]

#df = pd.read_csv('train_FD001.csv', sep=' ')

#Make a copy of DF
df_tr = df

#Transsform the timeOfDay to dummies
#df_tr = pd.get_dummies(df_tr, columns=['timeOfDay'])

# Add labels to Dataset. 
cl = np.linspace(0,10,len(df))
cl = np.rint(cl)
df_tr['labels'] = cl
#Standardize
clmns = ['S2', 'S3', 'S4', 'S7', 'S9', 'S11', 'S12','S15', 'S20', 'S21']
df_tr_std = stats.zscore(df_tr[clmns])



#Cluster the data
kmeans = KMeans(n_clusters=5, random_state=None).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to originaal data
df_tr['clusters'] = labels

#Add the column into our list
clmns.extend(['clusters'])

plt.scatter(range(len(df)),df_tr['clusters'])

#Lets analyze the clusters
#print df_tr[clmns].groupby(['clusters']).mean()
