#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:51:45 2022

@author: goharshoukat

Script does the following:
    1. Reloads models saved in a specific folder
    2. model evaluates data from input of test data
    3. Plots the data predictions
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model_dir = os.listdir('Models')
model_dir.remove('.DS_Store') # remove hidden fiel from directory

cols = ['Depth', 'Cone Resistance qc', 'Sleeve Friction fs']
test_files = pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()

test_data_dict = {} #cache all test file data in a 
#load the cpt data for each test file
data_dir = 'datasets/cpt_reformatted_datasets/'
for file in test_files.test:
    test_data_dict[file] = pd.read_csv(data_dir + file)

#generate the MinMaxScaler from the trained data to fit this on to the testing
#data. creating a new scalar would add discrepancies and false predictions
train = pd.read_csv('datasets/train.csv', dtype = str)
train = train.astype('float64')

#trainX contains only the input for the trainer which will be used to
#scale the test file inputs

trainX = np.array(train[['Depth', 'lat', 'lng', 'bathymetry']].copy())
scalar_trainer_X = MinMaxScaler(feature_range=(0,1))
trainX  = scalar_trainer_X.fit_transform(trainX)

#obtain the scalar for the training Y values
scalar_trainer_Y = MinMaxScaler(feature_range=(0,1))
trainY = np.array(train[['Cone Resistance qc', 'Sleeve Friction fs']].copy())
trainY = scalar_trainer_Y.fit_transform(trainY)


#Each ml model in the output will have 5 files reconstructed
for ml_model in model_dir:
    model = tf.keras.models.load_model('Models/' + ml_model)
    outdir = 'output/' + ml_model
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    for file in test_data_dict:
        df = test_data_dict[file]
        tX = np.array(df[['Depth', 'latitude', 'longitude', 'bathymetry']].copy())
        tX = scalar_trainer_X.transform(tX)
        tX1 = tX[:, 0]       #depth
        tX2 = tX[:, 1]       #lat
        tX3 = tX[:, 2]       #lng
        tX4 = tX[:, 3]       #bathy
    
        results = model.predict(
        {
        'depth' : tX1, 'lat' : tX2, 'lng' : tX3,
        'bathymetry' : tX4
        })
        results = np.array(results)
        results = np.transpose(results[:,:,0])
        #rescale the test values as per the previous scalar_trainer_y
        results = scalar_trainer_Y.inverse_transform(results)
        df2 = pd.DataFrame({'depth' : df['Depth'] ,
            'latitude' : df['latitude'], 'longitude' : df['longitude'],
            'qc' : results[:,0], 'fs':results[:,1]})
        
    
        df2.to_csv(outdir + '/reconstructed_{}.csv'.format(file[:-4]), index = False)
