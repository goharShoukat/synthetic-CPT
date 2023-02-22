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

#Test data
df = pd.DataFrame()
for slope, dropout, attempt in zip(slopes, dropouts, attempts):        
    for ml_model, depth in zip(model_dir, depths):
        model = r'output/Model Evaluation/{} Attempt/test/'.format(attempt) + ml_model +'/reconstructed_'
        
        for file in test_data_dict:
            df2 = pd.read_csv(model + file)
            df2['true qt'] = test_data_dict[file]['Smooth qt']
            df2['true fs'] = test_data_dict[file]['Smooth fs']
            df2['Attempt'] = attempt
            df2['model'] = ml_model
            df2['depth'] = depth
            df2['loc'] = file
            df2['slope'] = slope
            df2['dropout'] = dropout
            df2['classification'] = 'test'
            df = pd.concat([df, df2])
df = df.reset_index(drop=True)
df.to_csv('output/Reconstructed Combined/test.csv')
df1 = df #too combine with training later

#training data
df = pd.DataFrame()
for slope, dropout, attempt in zip(slopes, dropouts, attempts):        
    for ml_model, depth in zip(model_dir, depths):
        model = r'output/Model Evaluation/{} Attempt/'.format(attempt) + ml_model +'/reconstructed_'
        
        for file in train_data_dict:
            df2 = pd.read_csv(model + file)
            df2['true qt'] = train_data_dict[file]['Smooth qt']
            df2['true fs'] = train_data_dict[file]['Smooth fs']
            df2['Attempt'] = attempt
            df2['model'] = ml_model
            df2['depth'] = depth
            df2['loc'] = file
            df2['slope'] = slope
            df2['dropout'] = dropout
            df2['classification'] = 'train'
            df = pd.concat([df, df2])
df = df.reset_index(drop=True)
df.to_csv('output/Reconstructed Combined/train.csv')


#combined test-training
combined=pd.concat([df, df1])
combined.to_csv('output/Reconstructed Combined/combined.csv')