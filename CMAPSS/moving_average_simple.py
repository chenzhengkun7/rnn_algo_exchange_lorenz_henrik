# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:06:00 2018

@author: henri
"""

from pandas import Series
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


names=['EngNo', 'Cycle', 'OC1', 'OC2', 'OC3', 'S1', 'S2', 'S3',
       'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13',
       'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']

# load data
dataset = pd.read_csv('train_FD001.csv', sep=' ', names=names)
df = pd.DataFrame(dataset)
#print(df)
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
    rm = rm.drop(rm.index[0:(window_size-1)])
    data = pd.concat([data, rm])


    #plt.plot()

plt.plot(dataset['S9'])
plt.plot(data['S9'])
plt.show()

data.to_csv('train_FD001_filt.csv', encoding='utf-8', index=None, sep=' ')