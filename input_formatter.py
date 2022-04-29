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

direc = 'cpt_raw_data/'
files = (glob(direc+ '*.csv')) 
files = np.sort([x.replace('cpt_raw_data/', '') for x in files])
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'
del tmp
# %% file seperation
#60 - 20 - 20 for train-validation-test split

#60-40 split between 
train, test_tmp = train_test_split(files, test_size = 0.3, train_size=0.7) #split the data files into train, validate, test sets
test, validation = train_test_split(test_tmp, test_size = 0.5, train_size=0.5)

del test_tmp

def noise(signal, mu = 0, sigma = 0.1, factor = 1):
    #takes a signal and adds gaussian noise
    #outputs the signal with noise of the same length
    n = np.random.normal(mu, sigma, len(signal)) # noise normal distribution
    return( signal + n * factor) #add noise to clean signal
    
cols = ['Depth', 'Cone Resistance qc', 'Sleeve Friction fs']

#prepare training dataset
train_df = pd.DataFrame()
for f in train:
    df = pd.read_csv(direc + f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(direc + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()
    
    df['bathymetry'] = noise(np.ones(len(df)) * float(p_data.loc['Water Depth', 1]))
    df['lat'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']), factor = 1E-6)
    df['lng'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']), factor = 1E-4)
    train_df = train_df.append(df)
        
train_df = train_df.reset_index(drop = True)
train_df.to_csv('datasets/train.csv', index=False)

#prepare validdation dataset
validation_df = pd.DataFrame()
for f in validation:
    df = pd.read_csv(direc + f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(direc + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()
    
    df['bathymetry'] = noise(np.ones(len(df)) * float(p_data.loc['Water Depth', 1]))
    df['lat'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']), factor = 1E-7)
    df['lng'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']), factor = 1E-5)
    validation_df = validation_df.append(df)
        
validation_df = validation_df.reset_index(drop = True)
validation_df.to_csv('datasets/validation.csv', index=False)

#prepare test dataset
test_df = pd.DataFrame()
for f in test:
    df = pd.read_csv(direc + f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(direc + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()
    
    df['bathymetry'] = noise(np.ones(len(df)) * float(p_data.loc['Water Depth', 1]))
    df['lat'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']), factor = 1E-7)
    df['lng'] = noise(np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']), factor = 1E-5)
    test_df = test_df.append(df)
        
test_df = test_df.reset_index(drop = True)
test_df.to_csv('datasets/test.csv', index=False)