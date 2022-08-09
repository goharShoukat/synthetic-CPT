#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: goharshoukat

This script allows the fit to continue from the last saved checkpoint
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.models import load_model
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from customCallBack import PlotLearning
from model_definition import model_definition

train = pd.read_csv('datasets/train.csv', dtype = str)
train = train.astype('float64')
X = np.array(train[['Depth', 'lat', 'lng', 'bathymetry']].copy())
Y = np.array(train[['Cone Resistance qc', 'Sleeve Friction fs']].copy())
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

new_model = load_model("/Users/goharshoukat/Documents/GitHub/synthetic-CPT/Models/Fifth Attempt/Scaled/Model4_opt_adam_activation_LeakyReLU")
