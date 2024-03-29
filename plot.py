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
<<<<<<< HEAD
#and remove the unnecessary folders in the directory.
#Thirteenth reconstruct results in test directory and then in the training dir
reconst_model_test_dir = os.listdir('output/Model Evaluation/Thirteenth Attempt/test')
=======
#and remove the unnecessary folders in the directory. 
#first reconstruct results in test directory and then in the training dir
reconst_model_test_dir = os.listdir('output/Model Evaluation/Seventh Attempt/test')
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
if '.DS_Store' in reconst_model_test_dir:
   reconst_model_test_dir.remove('.DS_Store')


test_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_test_dir:
<<<<<<< HEAD
    files = os.listdir(r'output/Model Evaluation/Thirteenth Attempt/test/' + path)

    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Thirteenth Attempt/test/' + path):
        os.mkdir(r'output/Model Evaluation/Thirteenth Attempt/test/' + path)

    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:
        df = pd.read_csv(r'output/Model Evaluation/Thirteenth Attempt/test/' + path + '/'+ file)
=======
    files = os.listdir(r'output/Model Evaluation/Seventh Attempt/test/' + path)
    
    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Seventh Attempt/test/' + path):
        os.mkdir(r'output/Model Evaluation/Seventh Attempt/test/' + path)
        
    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:    
        df = pd.read_csv(r'output/Model Evaluation/Seventh Attempt/test/' + path + '/'+ file)
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
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
    #create one plot. for each plot, plot the 5 models and compare results
    
    fig, ax = plt.subplots(figsize = (30,30))
    for model in reconstructed:
        #run a for loop to compare each of the test data files reconstructed
        #using the models

        df_rec = reconstructed[model]['reconstructed_' + test_file[0]]
        ax.plot(df_rec['qc'], df_rec.depth, label = model, alpha = 0.3)
    
    ax.plot(df_orig['Cone Resistance qc'], df_orig.Depth, label = 'Original')
    ax.set_xlabel(r'Cone Resistance $Q_c$')
    ax.set_ylabel(r'Depth (m)')
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    #ax.set_title(file + '\n' + 'Lat: {}, Long: {}'.format(
        #    location.loc[location['CPT']==file[:-4], 'lat'].iloc[0],
        #    location.loc[location['CPT']==file[:-4], 'lng'].iloc[0]))
    plt.legend()
    
    #make output directory for testing data
<<<<<<< HEAD
    if not os.path.isdir(r'output/Model Evaluation/Thirteenth Attempt/plots/test/'):
        os.makedirs(r'output/Model Evaluation/Thirteenth Attempt/plots/test/')
    plt.savefig(r'output/Model Evaluation/Thirteenth Attempt/plots/test/' + str(test_file[0][:-4]) + '.pdf')
=======
    if not os.path.isdir(r'output/Model Evaluation/Seventh Attempt/plots/test/'):
        os.makedirs(r'output/Model Evaluation/Seventh Attempt/plots/test/')
    plt.savefig(r'output/Model Evaluation/Seventh Attempt/plots/test/' + str(test_file[0][:-4]) + '.pdf')
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
    plt.close()
    
###############################################################################

# %% read output from modelled data
<<<<<<< HEAD
#and remove the unnecessary folders in the directory.
#Thirteenth reconstruct results in test directory and then in the training dir
#same as above but for training data
reconst_model_train_dir = os.listdir('output/Model Evaluation/Thirteenth Attempt/')
=======
#and remove the unnecessary folders in the directory. 
#first reconstruct results in test directory and then in the training dir
#same as above but for training data
reconst_model_train_dir = os.listdir('output/Model Evaluation/Seventh Attempt/')
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
if '.DS_Store' in reconst_model_train_dir:
   reconst_model_train_dir.remove('.DS_Store')
if 'plots' in reconst_model_train_dir:
   reconst_model_train_dir.remove('plots')
if 'test' in reconst_model_train_dir:
   reconst_model_train_dir.remove('test')

train_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['train']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_train_dir:
<<<<<<< HEAD
    files = os.listdir(r'output/Model Evaluation/Thirteenth Attempt/' + path)

    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Thirteenth Attempt/' + path):
        os.mkdir(r'output/Model Evaluation/Thirteenth Attempt/' + path)

    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:
        df = pd.read_csv(r'output/Model Evaluation/Thirteenth Attempt/' + path + '/'+ file)
=======
    files = os.listdir(r'output/Model Evaluation/Seventh Attempt/' + path)
    
    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Seventh Attempt/' + path):
        os.mkdir(r'output/Model Evaluation/Seventh Attempt/' + path)
        
    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:    
        df = pd.read_csv(r'output/Model Evaluation/Seventh Attempt/' + path + '/'+ file)
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
        reconstructed[path][file] = df

###############################################################################
orig_direc = 'datasets/cpt_reformatted_datasets/'
orig_files = (glob(orig_direc+ '*.csv'))
orig_files = np.sort([x.replace('datasets/cpt_reformatted_datasets/', '') for x in orig_files])        

for train_file in train_files:
    print(train_file[0])
        #load reformatted cpt data files
    #if str(orig_file) in str(test_files):
    df_orig = pd.read_csv('datasets/cpt_reformatted_datasets/' + train_file[0])
    fig, ax = plt.subplots(figsize = (30,30))
    
    #compare each original datafile witht he data generated from the
    #several models. a for loop to run through each of the models
    for model in reconstructed:
        
        #run a for loop to compare each of the test data files reconstructed
        #using the models

        df_rec = reconstructed[model]['reconstructed_' + train_file[0]]
        
        #start plotting. both dataframes loaded
        ax.plot(df_rec['qc'], df_rec.depth, label = model, alpha = 0.3)
    ax.plot(df_orig['Cone Resistance qc'], df_orig.Depth, label = 'Original')
    ax.set_xlabel(r'Cone Resistance $Q_c$')
    ax.set_ylabel(r'Depth (m)')
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    plt.legend()
    #ax.set_title(file + '\n' + 'Lat: {}, Long: {}'.format(
    #    location.loc[location['CPT']==file[:-4], 'lat'].iloc[0],
    #    location.loc[location['CPT']==file[:-4], 'lng'].iloc[0]))
<<<<<<< HEAD
    if not os.path.isdir(r'output/Model Evaluation/Thirteenth Attempt/plots/train/'):
        os.makedirs(r'output/Model Evaluation/Thirteenth Attempt/plots/train/')

    plt.savefig(r'output/Model Evaluation/Thirteenth Attempt/plots/train/'  + str(train_file[0][:-4]) + '.pdf')
=======
    if not os.path.isdir(r'output/Model Evaluation/Seventh Attempt/plots/train/'):
        os.makedirs(r'output/Model Evaluation/Seventh Attempt/plots/train/')
    
    plt.savefig(r'output/Model Evaluation/Seventh Attempt/plots/train/'  + str(train_file[0][:-4]) + '.pdf')
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)
    plt.close()
        
#%% plot for all the cpt profiles
<<<<<<< HEAD
# =============================================================================
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# files = os.listdir('datasets/cpt_filtered_datasets/')
# fig, ax = plt.subplots(figsize = (30,30))
#
# for f in files:
#     df = pd.read_csv('datasets/cpt_filtered_datasets/' + f)
#
#
#     ax.plot(df['Cone Resistance qc'], df.Depth, linewidth = 0.5, alpha = 0.5,
#             label = f[:-4])
# ax.set_xlabel(r'Cone Resistance $q_c$')
# ax.set_ylabel(r'Depth (m)')
#
# ax.invert_yaxis()
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
# ax.grid()
# plt.show()
# ax.legend(loc='upper center',
#           ncol=6)
# plt.savefig('output/Original CPT Profiles/CPT combined.pdf')
#
# =============================================================================
=======
>>>>>>> parent of 132c5c3 (made changes to trainer algorithm.)

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
