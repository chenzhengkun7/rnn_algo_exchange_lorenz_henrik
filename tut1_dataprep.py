# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:23:41 2018

@author: henri

KERAS tutorial - China air pollution forecast
"""

import pandas as pd
import datetime as dt

#Load data

def parse(x):
    return dt.datetime.strptime(x, '%Y %m %d %H')

dataset = pd.read_csv('raw.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)

dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

dataset['pollution'].fillna(0, inplace=True)

dataset = dataset[24:]

print(dataset.head(5))

dataset.to_csv('pollution.csv')