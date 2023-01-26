#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:33:41 2022

@author: goharshoukat

Script to generate input and output for the ANN

"""

import pandas as pd
import numpy as np
from glob import glob
import random
from sklearn.model_selection import train_test_split

direc = 'datasets/cpt_filtered_datasets/'
files = (glob(direc+ '*.csv'))
files = np.sort([x.replace('datasets/cpt_filtered_datasets/', '') for x in files])
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'
del tmp




bathyFilesDirec = 'datasets/cpt_raw_data/'
bathyFiles = glob(bathyFilesDirec +'*.csv')
bathyFiles = np.sort([x.replace('datasets/cpt_raw_data/', '') for x in files])

# %% file seperation

#80-20 split between
#train, test = train_test_split(files, test_size = 0.2, train_size=0.8) #split the data files into train, validate, test sets
train = np.array([files[-1], files[21], files[20], files[19], files[18], files[17],
                  files[14], files[10], files[8], files[7], files[11], files[12],
                  files[0], files[15], files[1]])
test = np.array([files[22],  files[16], files[9], files[13]])




def noise(signal, mu = 0, sigma = 0.1, factor = 1):
    #takes a signal and adds gaussian noise
    #outputs the signal with noise of the same length
    n = np.random.normal(mu, sigma, len(signal)) # noise normal distribution
    return( signal + n * factor) #add noise to clean signal

cols = ['Depth', 'Corrected Cone Resistance qt', 'Sleeve Friction fs']

#prepare training dataset
train_df = pd.DataFrame()
for f in train:
    df = pd.read_csv(direc + f, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(bathyFilesDirec + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()

    df['bathymetry'] = noise(np.ones(len(df)) * float(p_data.loc['Water Depth', 1]))
    df['lat'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']), factor = 1E-8)
    df['lng'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']), factor = 1E-10)
    train_df = train_df.append(df)

train_df = train_df.reset_index(drop = True)
train_df['bathymetry'] = np.abs(train_df['bathymetry'])
train_df.to_csv('datasets/train.csv', index=False)

#prepare test dataset
test_df = pd.DataFrame()
for f in test:
    df = pd.read_csv(direc + f, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(bathyFilesDirec + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()

    df['bathymetry'] = noise(np.ones(len(df)) * float(p_data.loc['Water Depth', 1]))
    df['lat'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']), factor = 1E-7)
    df['lng'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']), factor = 1E-5)
    test_df = test_df.append(df)

test_df = test_df.reset_index(drop = True)
test_df.to_csv('datasets/test.csv', index=False)


#creat a summary file of the training and test data
for i in range(len(train) - len(test)):
    test = np.append(test, np.NAN)
summary = pd.DataFrame({'train' : train, 'test' : test})
summary.to_csv('datasets/summary.csv')
