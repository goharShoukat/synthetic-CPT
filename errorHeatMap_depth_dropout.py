#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 09:11:12 2022

@author: goharshoukat

script to determine the MSE between training-predicted value and test-predicted value

script not automated yet. manually done. 
"""

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import glob
#attempts = ['Twentythird', 'Twentyfourth', 'Twentyfifth']
#dropouts = ['0.2', '0.3', '0.1']


attempts = ['Twentyfifth', 'Twentythird', 'Twentyfourth', 'Twentysixth']
dropouts = ['0.1', '0.2', '0.3', '0.4']
models = np.arange(1, 6, 1).astype(str)
depths = list(range(1, 10)[::2])[::-1]
files = sorted(glob.glob('datasets/cpt_filtered_datasets/*.csv'))
train = sorted([files[-1], files[21], files[20], files[19], files[18], files[17],
                  files[14], files[13], files[10], files[8], files[7], files[11], files[12],
                  files[0], files[15]])
test = sorted([files[22],  files[16], files[1], files[9]])


trainMSE = pd.DataFrame(columns = ['Attempt', 'Model', 'depth', 'dropout','Location', 'MSE'])
trainAvg = pd.DataFrame(columns = ['depth', 'dropout', 'Avg MSE'])
avgTr = pd.DataFrame({'depth' : [0], 'dropout' : [0], 'Avg MSE' : [0]}) #dummy df

testMSE = pd.DataFrame(columns = ['Attempt', 'Model', 'depth', 'dropout','Location', 'MSE'])
testAvg = pd.DataFrame(columns = [ 'depth', 'dropout', 'Avg MSE'])
avgTe = pd.DataFrame({'depth' : [0], 'dropout' : [0], 'Avg MSE' : [0]}) #dummy df


for attempt, dropout in zip(attempts, dropouts):
    for model, depth in zip(models, depths):
        reconsTrain1 = sorted(glob.glob('output/Model Evaluation/{} Attempt/Model{}_opt_ADAM_activation_LeakyReLU/*.csv'.format(attempt, model)))
        reconsTest1 = sorted(glob.glob('output/Model Evaluation/{} Attempt/test/Model{}_opt_ADAM_activation_LeakyReLU/*.csv'.format(attempt, model)))
        
        df = pd.DataFrame({'Attempt' : attempt, 'Model' : model, 'depth' : depth, 'dropout' : dropout,
                                 'Location' : train, 'MSE' : np.nan, 
                                })
        
        for i in range(len(train)):
            
        
            reconsFile = pd.read_csv(reconsTrain1[i]).dropna()
            origFile = pd.read_csv(train[i])[:len(reconsFile)]
            df.loc[i, 'MSE'] = mean_squared_error(origFile['Cone Resistance qc'], reconsFile['qc'])
        trainMSE = pd.concat([trainMSE, df], axis = 0).reset_index(drop=True)
        avgTr['Avg MSE'] = df['MSE'].mean()
        avgTr[['depth', 'dropout']] = depth, dropout
        trainAvg = pd.concat([trainAvg, avgTr], axis = 0).reset_index(drop = True)        
        
        
        
        
        df2 = pd.DataFrame({'Attempt' : attempt, 'Model' : model, 'depth' : depth, 'dropout' : dropout,
                                 'Location' : test, 'MSE' : np.nan})
        
        for i in range(len(test)):
            
        
            reconsFile = pd.read_csv(reconsTest1[i]).dropna()
            origFile = pd.read_csv(test[i])[:len(reconsFile)]
            df2.loc[i, 'MSE'] = mean_squared_error(origFile['Cone Resistance qc'], reconsFile['qc'])
        
        testMSE = pd.concat([testMSE, df2], axis = 0).reset_index(drop=True)
        avgTe['Avg MSE'] = df2['MSE'].mean()
        avgTe[['depth', 'dropout']] = depth, dropout
        testAvg = pd.concat([testAvg, avgTe], axis = 0).reset_index(drop = True)        



#%%     
# =============================================================================
# Heat Map from the df generated above for the MSEs 
# this section discovers the best leak and depth ratio
# =============================================================================



import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 18})
#invert the depth vector for plotting reasons
depths = depths[::-1]
dropouts = dropouts[::-1]
De, Dr = np.meshgrid(depths, np.float32(dropouts))
Z_train = np.zeros(np.shape(De)) * np.nan
Z_test = np.zeros(np.shape(De)) * np.nan

for i in range(len(dropouts)):
    for j in range(len(depths)):
        Z_train[i, j] = trainAvg[(trainAvg['depth'] == depths[j]) & (trainAvg['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]
        Z_test[i, j]  = testAvg[(testAvg['depth'] == depths[j]) & (testAvg['dropout'] == dropouts[i])]['Avg MSE'].iloc[0]   


#fig, ax = plt.subplots(1, figsize=(30,30))        
sTrain = sns.heatmap(Z_train, vmin = np.min(Z_train), linecolor='white', 
                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                     square = True, vmax = 0.5, xticklabels=depths, 
                     yticklabels=dropouts,)

sTrain.set(ylabel='Dropout Probability', xlabel = 'Depth')
#plt.title('a) Training Dataset')
plt.savefig('output/plots/train_depth_dropout.png', dpi = 200)
plt.close()

#fig, ax = plt.subplots(1, figsize=(30,30))       
sTest = sns.heatmap(Z_test, vmin = np.min(Z_test), linecolor='white', 
                     linewidths=0.5, cbar_kws={'label': 'MSE'}, annot=True, 
                     square = True, vmax = 0.7, xticklabels=depths, 
                     yticklabels=dropouts, )#, fmt = "0.3g")
sTest.set(ylabel='Dropout Probability', xlabel = 'Depth')
#plt.title('b) Testing Dataset')
plt.savefig('output/plots/test_depth_dropout.png', dpi = 200)
plt.close()


# =============================================================================
# This section discovers the best dropout to depth ratio
# =============================================================================


# =============================================================================
# This section discovers best dropout to leak ratio
# =============================================================================
