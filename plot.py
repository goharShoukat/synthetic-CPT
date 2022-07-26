#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:47:52 2022

@author: goharshoukat

script to generate plots of the original files

Generate plots from original CPT data
"""
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
##############################################################################

# %% read output from modelled data
#and remove the unnecessary folders in the directory. 
#first reconstruct results in test directory and then in the training dir
reconst_model_test_dir = os.listdir('output/Model Evaluation/Third Attempt/test')
if '.DS_Store' in reconst_model_test_dir:
   reconst_model_test_dir.remove('.DS_Store')


test_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_test_dir:
    files = os.listdir(r'output/Model Evaluation/Third Attempt/test/' + path)
    
    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Third Attempt/test/' + path):
        os.mkdir(r'output/Model Evaluation/Third Attempt/test/' + path)
        
    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:    
        df = pd.read_csv(r'output/Model Evaluation/Third Attempt/test/' + path + '/'+ file)
        reconstructed[path][file] = df

###############################################################################
orig_direc = 'datasets/cpt_reformatted_datasets/'
orig_files = (glob(orig_direc+ '*.csv'))
orig_files = np.sort([x.replace('datasets/cpt_reformatted_datasets/', '') for x in orig_files])        

for test_file in test_files:
    print(test_file[0])
    #load reformatted cpt data files
    #if str(orig_file) in str(test_files):
    df_orig = pd.read_csv('datasets/cpt_reformatted_datasets/' + test_file[0])
    
    #compare each original datafile witht he data generated from the
    #several models. a for loop to run through each of the models
    for model in reconstructed:
        
        #run a for loop to compare each of the test data files reconstructed
        #using the models

        df_rec = reconstructed[model]['reconstructed_' + test_file[0]]
        
        #start plotting. both dataframes loaded
        fig, ax = plt.subplots(figsize = (30,30))
        ax.plot(df_orig['Cone Resistance qc'], df_orig.Depth, label = 'Original')
        ax.plot(df_rec['qc'], df_rec.depth, label = 'Reconst.')
        ax.set_xlabel(r'Cone Resistance $Q_c$')
        ax.set_ylabel(r'Depth (m)')
        ax.invert_yaxis()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.grid()
        #ax.set_title(file + '\n' + 'Lat: {}, Long: {}'.format(
        #    location.loc[location['CPT']==file[:-4], 'lat'].iloc[0],
        #    location.loc[location['CPT']==file[:-4], 'lng'].iloc[0]))
        plt.savefig(r'output/Model Evaluation/Third Attempt/test/' + model + str(test_file[0][:-4]) + '.pdf')
        plt.close()
        
###############################################################################

# %% read output from modelled data
#and remove the unnecessary folders in the directory. 
#first reconstruct results in test directory and then in the training dir
#same as above but for training data
reconst_model_train_dir = os.listdir('output/Model Evaluation/Third Attempt/')
if '.DS_Store' in reconst_model_train_dir:
   reconst_model_train_dir.remove('.DS_Store')
if 'test' in reconst_model_train_dir:
   reconst_model_train_dir.remove('test')


train_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_train_dir:
    files = os.listdir(r'output/Model Evaluation/Third Attempt/' + path)
    
    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Third Attempt/' + path):
        os.mkdir(r'output/Model Evaluation/Third Attempt/' + path)
        
    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:    
        df = pd.read_csv(r'output/Model Evaluation/Third Attempt/' + path + '/'+ file)
        reconstructed[path][file] = df

###############################################################################
orig_direc = 'datasets/cpt_reformatted_datasets/'
orig_files = (glob(orig_direc+ '*.csv'))
orig_files = np.sort([x.replace('datasets/cpt_reformatted_datasets/', '') for x in orig_files])        

for orig_file in orig_files:
        #load reformatted cpt data files
    #if str(orig_file) in str(test_files):
    df_orig = pd.read_csv('datasets/cpt_reformatted_datasets/' + orig_file)
    
    #compare each original datafile witht he data generated from the
    #several models. a for loop to run through each of the models
    for model in reconstructed:
        
        #run a for loop to compare each of the test data files reconstructed
        #using the models

        df_rec = reconstructed[model]['reconstructed_' + orig_file]
        
        #start plotting. both dataframes loaded
        fig, ax = plt.subplots(figsize = (30,30))
        ax.plot(df_orig['Cone Resistance qc'], df_orig.Depth, label = 'Original')
        ax.plot(df_rec['qc'], df_rec.depth, label = 'Reconst.')
        ax.set_xlabel(r'Cone Resistance $Q_c$')
        ax.set_ylabel(r'Depth (m)')
        ax.invert_yaxis()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.grid()
        #ax.set_title(file + '\n' + 'Lat: {}, Long: {}'.format(
        #    location.loc[location['CPT']==file[:-4], 'lat'].iloc[0],
        #    location.loc[location['CPT']==file[:-4], 'lng'].iloc[0]))
        plt.savefig(r'output/Model Evaluation/Third Attempt/test/' + model + str(orig_file[:-4]) + '.pdf')
        plt.close()
        
#%% plot for all the cpt profiles

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
files = os.listdir('datasets/cpt_reformatted_datasets/')
fig, ax = plt.subplots(figsize = (30,30))

for f in files:
    df = pd.read_csv('datasets/cpt_reformatted_datasets/' + f)
    
    
    ax.plot(df['Cone Resistance qc'], df.Depth, linewidth = 0.5, alpha = 0.5,
            label = f[:-4])
ax.set_xlabel(r'Cone Resistance $q_c$')
ax.set_ylabel(r'Depth (m)')
   
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
plt.show()
ax.legend(loc='upper center',
          ncol=6)
plt.savefig('output/Original CPT Profiles/CPT combined.pdf')
