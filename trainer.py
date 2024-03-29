#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:31:30 2022

@author: goharshoukat

first model attempt

Only to understand how to use tensorflow
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from customCallBack import PlotLearning
from model_definition import model_definition
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
###############################################################################
#import training dataset
train = pd.read_csv('datasets/train.csv', dtype = str)
train = train.astype('float64')
X = np.array(train[['Depth', 'lat', 'lng', 'bathymetry']].copy())
Y = np.array(train[['Smooth qt', 'Smooth fs']].copy())
X, Y = shuffle(X, Y)
###############################################################################

# %% X and Y are scaled as per their independent
scalarX = MinMaxScaler(feature_range=(0,1))
X  = scalarX.fit_transform(X)


#scale tY wrt input Y matrix
scalarY = MinMaxScaler(feature_range=(0, 1))
Y = scalarY.fit_transform(Y)

###############################################################################
#training data
# divide all the inputs into seperate vectors
X1 = X[:, 0]       #depth
X2 = X[:, 1]       #lat
X3 = X[:, 2]       #lng
X4 = X[:, 3]       #bathy

Y1 = Y[:, 0]       #qc
Y2 = Y[:, 1]       #fs


###############################################################################
#create the 4 different inputs that will feed into the model
input1 =  keras.Input(shape=(1,), name = 'depth')
input2 = keras.Input(shape=(1,), name = 'lat')
input3 = keras.Input(shape = (1,), name = 'lng')
input4 = keras.Input(shape=(1, ), name = 'bathymetry')

#merge the 4 inputs into one vector for matrix multiplication
merge = layers.Concatenate(axis = 1)([input1, input2, input3, input4])


#list of nodes in the model.

#call the model_definition function which has 4 different models
#the model also has optimizer information
model_def = model_definition()['models']
optim     = model_definition()['optimizers']
#optim = ['adam']
activationFunc = ['LeakyReLU']
attempt = 'Nineth' #quantifies the different tweaks made.

#for activation in activationFunc:
#    for o in optim:
for mod in model_def:
    #make output folder to house the model files
    model_dir = r"Models/{} Attempt/Scaled/{}_opt_ADAM_activation_LeakyReLU/".format(
                                    attempt, mod, 'LeakyReLU')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    n_nodes = model_def[mod] #variable to extract array with layers
    #implementation via for loops
    for index, nodes in enumerate(n_nodes):
        if index == 0:
            l = layers.Dense(nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(merge)
        #if index == 0 or index == 1 or index == 3 or index == 5 or index == 7:
        if index == 0 or index == 1:
            l = layers.Dropout(0.1)(l)
        l = layers.Dense(nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                         kernel_regularizer=regularizers.L1L2(l1=1e-6, l2=1e-6),
                bias_regularizer=regularizers.L1(1e-6),
                activity_regularizer=regularizers.L1(1e-6))(l)





    #create the 2 outputs for the last layer
    output1 = layers.Dense(1, activation='linear', name = 'qt')(l) #qc
    output2 = layers.Dense(1, activation='linear', name = 'fs')(l) #fs

    model = keras.Model(
        inputs = [input1, input2, input3, input4],
        outputs = [output1, output2]
    )

    keras.utils.plot_model(model, model_dir + 'model.pdf', show_shapes=True)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = {
            'qt' : keras.losses.MeanSquaredError(),
            'fs' : keras.losses.MeanSquaredError(),
        },
    )

    model.summary()

    #create directory to save checkpoints
    checkpoint_path = model_dir


    #save weights at the end of each epoch
    batch_size = 64
    plt_callback = PlotLearning(model_dir)
    model_save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        mode='min',
        save_best_only=True, period = 10)
    #provides access to loss etc
    #can be accessed via different keys
    #history.keys()
    history = model.fit(
        {
        'depth' : X1, 'lat' : X2, 'lng' : X3,
        'bathymetry' : X4
        },
        {
        'qt' : Y1,
        'fs' : Y2
        },
        validation_split = 0.1,

        batch_size=batch_size, epochs = 500, verbose=1, shuffle=True,
        callbacks=[plt_callback, model_save_callback]
    )