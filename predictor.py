#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:48:32 2022

@author: goharshoukat


Script does the following:
    1. Reloads models saved in a specific folder
    2. model predicts data for locations other than the ones in the 24 sites
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


# =============================================================================
# Prepare prediction location as per the required format
# =============================================================================
folder = 'Predictions I-More/'
file = pd.read_csv(folder + 'Synthetic_CPT_Locations_GM.csv')
file.columns = ['lng', 'lat', 'bathymetry', 'target', '']
file['bathymetry'] = np.abs(file['bathymetry'])

predict = {} #declare dictionary to hold all the dataframes
for i in range(len(file)):
    z = np.arange(0, file.loc[i,'target'], 0.01)
    df = pd.DataFrame({'Depth' : z, 'lat' : file.loc[i,'lat'], 
                   'lng' : file.loc[i, 'lng'], 'bathymetry' : (file.loc[i, 'bathymetry'])})
    predict['location {}'.format(i)] = df


# =============================================================================
# load files for scaling and rescaling transformations
# requires original training data since input output values are all scaled as 
# per training data    
# =============================================================================
train_files = pd.read_csv('datasets/summary.csv', usecols=['train'])
train_data_dict = {} #added to validate performance on training data
#load the cpt data for each test file
data_dir = 'datasets/cpt_filtered_datasets/'

for file in train_files.train:
    train_data_dict[file] = pd.read_csv(data_dir + file)
#above code on training data makes use of inidivual data files without noise
#the next few lines will make use of the entire training dataset with noise

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



#load model 4 of nineth attempt identified as best from hyperparameter tuning
model = tf.keras.models.load_model('/Users/goharshoukat/Documents/GitHub/synthetic-CPT/Models/Nineth Attempt/Scaled/Model4_opt_ADAM_activation_LeakyReLU')
outdir = r'Predictions I-More/'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

for file in predict:
    df = predict[file]
    tX = np.array(df[['Depth', 'lat', 'lng', 'bathymetry']].copy())
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
        'latitude' : df['lat'], 'longitude' : df['lng'], 'bathymetry' : df['bathymetry'],
        'qc' : results[:,0], 'fs':results[:,1]})


    df2.to_csv(outdir + '/{}.csv'.format(file), index = False)
    

