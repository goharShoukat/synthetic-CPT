#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:21:45 2022

@author: goharshoukat
"""

import pandas as pd
import glob
import numpy as np
from sklearn.metrics import mean_squared_error

summary = pd.read_excel('models/attempt_summary.xlsx', usecols=['Slope', 'Dropout'])
attempts = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 
            'Nineth', 'Tenth', 'Eleventh', 'Twelveth', 'Thirteenth']
dropouts  = summary['Dropout']
slopes = summary['Slope']
depths = np.array([1, 3, 5, 7, 9])
models = np.arange(1, 6, 1).astype(str)[::-1]

files = sorted(glob.glob('datasets/cpt_filtered_datasets/*.csv'))
train = sorted(np.array([files[-1], files[21], files[20], files[19], files[18], files[17],
                  files[14], files[10], files[8], files[7], files[11], files[12],
                  files[0], files[15], files[1]]))
test = sorted(np.array([files[22],  files[16], files[9], files[13]]))



trainMSE = pd.DataFrame(columns = ['Attempt', 'Model', 'depth', 'dropout', 'MSE'])
trainAvg = pd.DataFrame(columns = ['depth', 'dropout', 'Avg MSE'])
avgTr = pd.DataFrame({'depth' : [0], 'dropout' : [0], 'Avg MSE' : [0], 'slope' : [0]}) #dummy df


testMSE = pd.DataFrame(columns = ['Attempt', 'Model', 'depth', 'dropout','Location', 'MSE'])
testAvg = pd.DataFrame(columns = [ 'depth', 'dropout', 'Avg MSE'])
avgTe = pd.DataFrame({'depth' : [0], 'dropout' : [0], 'Avg MSE' : [0]}) #dummy df

for attempt, dropout, slope in zip(attempts, dropouts, slopes):
    for model, depth in zip(models, depths):
        reconsTrain1 = sorted(glob.glob('output/Model Evaluation/{} Attempt/Model{}_opt_ADAM_activation_LeakyReLU/*.csv'.format(attempt, model)))
        reconsTest1 = sorted(glob.glob('output/Model Evaluation/{} Attempt/test/Model{}_opt_ADAM_activation_LeakyReLU/*.csv'.format(attempt, model)))
        
        df = pd.DataFrame({'Attempt' : attempt, 'Model' : model, 'depth' : depth, 'dropout' : dropout,
                                 'Location' : train, 'MSE' : np.nan, 'slope' : np.nan
                                })
        
        for i in range(len(train)):
            
        
            reconsFile = pd.read_csv(reconsTrain1[i]).dropna()
            origFile = pd.read_csv(train[i])[:len(reconsFile)]
            df.loc[i, 'MSE'] = mean_squared_error(origFile['Sleeve Friction fs'], reconsFile['fs'])
        trainMSE = pd.concat([trainMSE, df], axis = 0).reset_index(drop=True)
        avgTr['Avg MSE'] = df['MSE'].mean()
        avgTr[['depth', 'dropout', 'slope']] = depth, dropout, slope
        trainAvg = pd.concat([trainAvg, avgTr], axis = 0).reset_index(drop = True)        
        
        
        
        df2 = pd.DataFrame({'Attempt' : attempt, 'Model' : model, 'depth' : depth, 'dropout' : dropout,
                                 'Location' : test, 'MSE' : np.nan})
        
        for i in range(len(test)):
            
        
            reconsFile = pd.read_csv(reconsTest1[i]).dropna()
            origFile = pd.read_csv(test[i])[:len(reconsFile)]
            df2.loc[i, 'MSE'] = mean_squared_error(origFile['Sleeve Friction fs'], reconsFile['qc'])
        
        testMSE = pd.concat([testMSE, df2], axis = 0).reset_index(drop=True)
        avgTe['Avg MSE'] = df2['MSE'].mean()
        avgTe[['depth', 'dropout', 'slope']] = depth, dropout, slope
        testAvg = pd.concat([testAvg, avgTe], axis = 0).reset_index(drop = True)        
# =============================================================================
# plotting training dataset for dropout against depth
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 10})

dropouts = [0.1, 0.2, 0.3, 0.4][::-1]
slopes = [0.1, 0.2, 0.3]
De, Dr = np.meshgrid(np.float32(depths), np.float32(dropouts))
Z_train = np.zeros(np.shape(De)) * np.nan
Z_test = np.zeros(np.shape(De)) * np.nan
f,(ax) = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,0.08]}, figsize=(30,30))
for k in range(len(slopes)):
        df01 = trainAvg[trainAvg['slope']==slopes[k]]
        for i in range(len(dropouts)):
            for j in range(len(depths)):
        
                Z_train[i, j] = df01[(df01['depth'] == depths[j]) & (df01['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]
        #        Z_test[i, j]  = testAvg[(testAvg['depth'] == depths[j]) & (testAvg['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]   
        #if k == len(slopes) - 1:
        cbarLogic = True
        '''
        sTrain = sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                 linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                 square = True, vmax = 0.5, xticklabels=depths, 
                                 yticklabels=dropouts, ax = ax[2], cbar=cbarLogic, 
                                 cbar_ax = ax[3])
        #else:
        '''
        if k < 2:
            cbarLogic = False
            sTrain = sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                 square = True, vmax = 309, xticklabels=depths, 
                                 yticklabels=dropouts, ax = ax[k], cbar= False)
        if k == 2:
            sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                     square = True, vmax = 309, xticklabels=depths, 
                                     yticklabels=dropouts, ax = ax[k], cbar_ax = ax[3])
        
        sTrain.set(ylabel='Dropout Probability', xlabel = 'Depth')

        ax[k].set_title('Slope = {}'.format(slopes[k]))
ax[0].get_shared_y_axes().join(ax[1],ax[2])
ax[1].set_ylabel('')
ax[1].set_yticks([])
ax[2].set_ylabel('')
ax[2].set_yticks([])
plt.savefig('output/training-depth-dropout-fs.png', transparent = True, dpi = 500)

# =============================================================================
# plotting testing dataset for dropout against depth 
# =============================================================================

dropouts = [0.1, 0.2, 0.3, 0.4][::-1]
slopes = [0.1, 0.2, 0.3]
De, Dr = np.meshgrid(np.float32(depths), np.float32(dropouts))
Z_train = np.zeros(np.shape(De)) * np.nan
Z_test = np.zeros(np.shape(De)) * np.nan
f,(ax) = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,0.08]}, figsize=(30,30))
for k in range(len(slopes)):
        df01 = testAvg[testAvg['slope']==slopes[k]]
        for i in range(len(dropouts)):
            for j in range(len(depths)):
        
                Z_train[i, j] = df01[(df01['depth'] == depths[j]) & (df01['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]
        #        Z_test[i, j]  = testAvg[(testAvg['depth'] == depths[j]) & (testAvg['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]   
        #if k == len(slopes) - 1:
        cbarLogic = True
        '''
        sTrain = sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                 linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                 square = True, vmax = 0.5, xticklabels=depths, 
                                 yticklabels=dropouts, ax = ax[2], cbar=cbarLogic, 
                                 cbar_ax = ax[3])
        #else:
        '''
        if k < 2:
            cbarLogic = False
            sTrain = sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                 square = True, vmax = 1570, xticklabels=depths, 
                                 yticklabels=dropouts, ax = ax[k], cbar= False)
        if k == 2:
            sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                                     square = True, vmax = 1570, xticklabels=depths, 
                                     yticklabels=dropouts, ax = ax[k], cbar_ax = ax[3])
        
        sTrain.set(ylabel='Dropout Probability', xlabel = 'Depth')

        ax[k].set_title('Slope = {}'.format(slopes[k]))
ax[0].get_shared_y_axes().join(ax[1],ax[2])
ax[1].set_ylabel('')
ax[1].set_yticks([])
ax[2].set_ylabel('')
ax[2].set_yticks([])
plt.savefig('output/plots/test_depth_dropout-fs.png', transparent = True, dpi = 500)