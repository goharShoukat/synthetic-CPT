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

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#import training dataset
df = pd.read_csv('datasets/train.csv', dtype = str)
df = df.astype('float64')
X = np.array(df[['Depth', 'lat', 'lng', 'bathymetry']].copy())
Y = np.array(df[['Cone Resistance qc', 'Sleeve Friction fs']].copy())
X, Y = shuffle(X, Y)

# %%
scalar = MinMaxScaler(feature_range=(0,1))
X  = scalar.fit_transform(X)
Y = scalar.fit_transform(Y)

# divide all the inputs into seperate vectors
X1 = X[:, 0]       #depth
X2 = X[:, 1]       #lat
X3 = X[:, 2]       #lng
X4 = X[:, 3]       #bathy

Y1 = Y[:, 0]       #qc
Y2 = Y[:, 1]       #fs

#create the 4 different inputs that will feed into the model
input1 =  keras.Input(shape=(1,), name = 'depth')
input2 = keras.Input(shape=(1,), name = 'lat')
input3 = keras.Input(shape = (1,), name = 'lng')
input4 = keras.Input(shape=(1, ), name = 'bathymetry')



#merge the 4 inputs into one vector for matrix multiplication
#2 hidden layers and one output layer
merge = layers.Concatenate(axis=1)([input1, input2, input3, input4])
l1 = layers.Dense(64, activation = 'relu')(merge)
l2 = layers.Dense(64, activation = 'relu')(l1)
l3 = layers.Dense(32, activation = 'relu')(l2)
l4 = layers.Dense(16, activation='relu')(l3)
l5 = layers.Dense(5, activation='relu')(l4)

l2 = layers.Dropout(0.5)(l2)

#create the 2 outputs for the last layer
output1 = layers.Dense(1, activation='linear', name = 'qc')(l5) #qc
output2 = layers.Dense(1, activation='linear', name = 'fs')(l5) #fs

model = keras.Model(
    inputs = [input1, input2, input3, input4],
    outputs = [output1, output2]
)

#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
model.compile(
    optimizer = 'adam',
    loss = {
        'qc' : keras.losses.MeanSquaredError(),
        'fs' : keras.losses.MeanSquaredError(),
    },
    metrics = ['mse']
)
model.summary()

#create directory to save checkpoints
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#save weights at the end of each epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 )

model.fit(
    {
    'depth' : X1, 'lat' : X2, 'lng' : X3,
    'bathymetry' : X4
    },

    {
    'qc' : Y1,
    'fs' : Y2
    },
    batch_size=64, epochs = 100, verbose=2, shuffle=True,callbacks=cp_callback

)
