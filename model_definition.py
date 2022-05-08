#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 09:52:02 2022

@author: goharshoukat

This script defines the different models and the two optimizers which will
be used


"""

import numpy as np
import tensorflow as tf
def model_definition():
    model1 = (np.linspace(8, 64, 8)).astype(int)
    model1 = np.append(model1, model1[::-1])

    model2 = np.linspace(8, 32, 4).astype(int)
    model2 = np.append(model2, model2[::-1])

    model3 = np.linspace(8, 16, 2)
    model3 = np.append(model3, model3[::-1])

    model4 = [8]
    models = {'Model1' : model1, 'Model2' : model2,
               'Model3' : model3, 'Model4' : model4}

    optimizers = ['adam', 'sgd']
    return {'models' : models, 'optimizers' : optimizers}
