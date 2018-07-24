# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:33:44 2018

@author: henri

Keras tutorial data plotting
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values

groups = [0,1,2,3,4,5,6,7]
i = 1

plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
    
plt.show()
