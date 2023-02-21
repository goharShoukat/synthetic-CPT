#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:19:09 2023

@author: goharshoukat

heatmap to display mse calculated using entire dataset of true values mapped again predicted values
"""

import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error


summary = pd.read_excel('hyperparameter_tuning.xlsx', usecols=['slope', 'dropout'])
attempts = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 
            'Nineth', 'Tenth', 'Eleventh', 'Twelfth', 'Thirteenth', 
            'Fourteenth', 'Fifteenth', 'Sixteenth']
dropouts  = summary['dropout']
slopes = summary['slope']
depths = np.array([1, 3, 5, 7, 9])[::-1]
#models = np.arange(1, 6, 1).astype(str)[::-1]

test_files = pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()
train_files = pd.read_csv('datasets/summary.csv', usecols=['train'])

test_data_dict = {} #cache all test file data in a
train_data_dict = {} #added to validate performance on training data
#load the cpt data for each test file
data_dir = 'datasets/cpt_filtered_datasets/'
for file in test_files.test:
    test_data_dict[file] = pd.read_csv(data_dir + file)

for file in train_files.train:
    train_data_dict[file] = pd.read_csv(data_dir + file)
    


model_dir = sorted(os.listdir(r'Models/Sixteenth Attempt/Scaled/'))
if '.DS_Store' in model_dir:
    model_dir.remove('.DS_Store') # remove hidden fiel from directory

df = pd.DataFrame()
for ml_model, depth in zip(model_dir, depths):
    model = r'output/Model Evaluation/Sixteenth Attempt/test/' + ml_model

    for file in test_data_dict:
        df2 = test_data_dict[file]
        #rescale the test values as per the previous scalar_trainer
        df2['model'] = ml_model
        df2['depth'] = depth
        df2['loc'] = file
        df = pd.concat([df, df2])
