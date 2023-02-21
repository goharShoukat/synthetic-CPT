#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:06:08 2022

@author: goharshoukat

filtering datasets for syntehtic cpt before feeding them into the 
ML algorithm
"""

import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
files = sorted(glob('datasets/cpt_reformatted_datasets_untouched/*.csv'))

# =============================================================================

# 
df = pd.read_csv(files[0])

plt.plot(df['Corrected Cone Resistance qt'], df.Depth)
# # =============================================================================
# # FIlteration using zscores > 3
# # =============================================================================
# levels = np.linspace(0, 25, 6)
# interv = df['Cone Resistance qc'].groupby(pd.cut(df.Depth, bins = levels)).aggregate([np.mean, np.std])
# 
# 
# #seperate the data based on the intervals
# data = {}
# for i in range(len(levels) - 1):
#     ser = [True if x in interv.index[i] else False for x in df['Depth']]
#     df2 = df[ser]
#     #df2['Cone Resistance qc'] = np.where(np.abs(zscore(df2['Cone Resistance qc'])) < 3, df2['Cone Resistance qc'], np.nan)
#     df2 = df2.mask(df2.sub(df2.mean()).div(df2.std()).abs().gt(2))
#     #df2 = df2[(np.abs(zscore(df2['Cone Resistance qc'])) < 3)]
#     df2 = df2.interpolate(method = 'linear')
#     data[interv.index[i]] = df2
#       
# 
# 
# #concetenate the 5 dfs together to create the final dataset
# df3 = data[interv.index[0]]
# for i in range(1, len(levels)-1):
#     df3 = pd.concat([df3, data[interv.index[i]]])
# 
# 
# 
# # =============================================================================
# # kernel smoothing 
# # =============================================================================
from statsmodels.nonparametric.kernel_regression import KernelReg

x = np.linspace(np.min(df.Depth), np.max(df.Depth), 200)



kr = KernelReg(endog = df['Corrected Cone Resistance qt'], exog = df.Depth, var_type = ['c'], bw = 'aic')
y_pred, std = kr.fit(x)

fig, ax = plt.subplots(figsize = (4.5, 10))
plt.plot
ax.plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 1, alpha = 0.5, label = 'Raw')
ax.plot( y_pred,x, label = 'Smoothed')

ax.set_xlabel(r'Corrected Cone Resistance $q_t (MPa)$')
ax.set_ylabel(r'Depth $(m)$')
ax.yaxis.labelpad = 0.05    
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.yaxis.set_label_coords(-0.1,0.5)
plt.show()
ax.legend(loc='upper center',
          ncol=6)
plt.savefig('output/plots/kernel_smoothed.png', dpi = 600, transparent = True)
# =============================================================================
levels = np.linspace(0, 25, 6)
interv = df['Cone Resistance qc'].groupby(pd.cut(df.Depth, bins = levels)).aggregate([np.mean, np.std])


#seperate the data based on the intervals
data = {}
for i in range(len(levels) - 1):
    ser = [True if x in interv.index[i] else False for x in df['Depth']]
    df2 = df[ser]
    #df2['Cone Resistance qc'] = np.where(np.abs(zscore(df2['Cone Resistance qc'])) < 3, df2['Cone Resistance qc'], np.nan)
    df2 = df2.mask(df2.sub(df2.mean()).div(df2.std()).abs().gt(2))
    #df2 = df2[(np.abs(zscore(df2['Cone Resistance qc'])) < 3)]
    df2 = df2.interpolate(method = 'linear')
    data[interv.index[i]] = df2
      


#concetenate the 5 dfs together to create the final dataset
df3 = data[interv.index[0]]
for i in range(1, len(levels)-1):
    df3 = pd.concat([df3, data[interv.index[i]]])



# =============================================================================
# kernel smoothing 
# =============================================================================
# =============================================================================
# from statsmodels.nonparametric.kernel_regression import KernelReg
# 
# x = np.linspace(np.min(df.Depth), np.max(df.Depth), 200)
# 
# 
# 
# kr = KernelReg(endog = df['Cone Resistance qc'], exog = df.Depth, var_type = ['c'])
# y_pred, std = kr.fit(x)
# 
# fig, ax = plt.subplots(figsize = (30,30))
# plt.plot
# ax.plot(df['Cone Resistance qc'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
# ax.plot( y_pred,x, label = 'filtered')
# ax.set_xlabel(r'Cone Resistance $q_c$')
# ax.set_ylabel(r'Depth (m)')
#    
# ax.invert_yaxis()
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
# ax.grid()
# 
# plt.show()
# ax.legend(loc='upper center',
#           ncol=6)
# # =============================================================================
# =============================================================================
# plots
# =============================================================================
from csaps import csaps
for f in files:
    df = pd.read_csv(f)
    x = np.linspace(np.min(df.Depth), np.max(df.Depth), len(df))
    qc_pred = csaps(df.Depth, df['Corrected Cone Resistance qt'], x, smooth=(0.5))
    fs_pred = csaps(df.Depth, df['Sleeve Friction fs'], x, smooth=(0.5))
    
    fig, ax = plt.subplots(figsize = (30,30))
    ax.plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
    ax.plot(qc_pred,x, label = 'filtered')
    ax.set_xlabel(r'Corrected Cone Resistance $q_t$')
    ax.set_ylabel(r'Depth (m)')
        
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    ax.legend(loc='upper center',
              ncol=6)    
    ax.set_title(f)
   # plt.show()



    fig, ax = plt.subplots(figsize = (30,30))
    ax.plot(df['Sleeve Friction fs'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
    ax.plot(fs_pred,x, label = 'filtered')
    ax.set_xlabel(r'Sleeve Friction $f_s$')
    ax.set_ylabel(r'Depth (m)')
        
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    ax.legend(loc='upper center',
              ncol=6)    
    ax.set_title(f)

#    plt.show()
    
    plt.show()
    
    
    df['Smooth qt'] = qc_pred
    df['Smooth fs'] = fs_pred
    f = f.replace("datasets/cpt_reformatted_datasets_untouched/", "datasets/cpt_filtered_datasets/")
    df.to_csv(f)
    
# =============================================================================
# adjust the smoothing factor for cpt 10 and 11 to get better fits 
# =============================================================================
df = pd.read_csv(files[9])
df = df.iloc[:760]
x = np.linspace(np.min(df.Depth), np.max(df.Depth), len(df))
qc_pred = csaps(df.Depth, df['Corrected Cone Resistance qt'], x, smooth=(.1))
fs_pred = csaps(df.Depth, df['Sleeve Friction fs'], x, smooth=(0.1))
fig, ax = plt.subplots(figsize = (30,30))
ax.plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
ax.plot(qc_pred,x, label = 'filtered')
ax.set_xlabel(r'Corrected Cone Resistance $q_t$')
ax.set_ylabel(r'Depth (m)')
    
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.legend(loc='upper center',
          ncol=6)    
ax.set_title(files[9])
plt.show()

fig, ax = plt.subplots(figsize = (30,30))

ax.plot(df['Sleeve Friction fs'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
ax.plot(fs_pred, x , label = 'filtered')
ax.set_xlabel(r'Sleeve Friction $f_s$')
ax.set_ylabel(r'Depth (m)')

ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.legend(loc='upper center',
          ncol=6)    
ax.set_title(files[9])
plt.show()



df['Smooth qt'] = qc_pred
df['Smooth fs'] = fs_pred
f = files[9]

f = f.replace("datasets/cpt_reformatted_datasets_untouched/", "datasets/cpt_filtered_datasets/")
df.to_csv(f)
# =============================================================================
# Remove the last data point from CPT09
# =============================================================================
df = pd.read_csv(files[7])
x = np.linspace(np.min(df.Depth), np.max(df.Depth), len(df))
qc_pred = csaps(df.Depth, df['Corrected Cone Resistance qt'], x, smooth=(0.5))
fs_pred = csaps(df.Depth, df['Sleeve Friction fs'], x, smooth=(0.5))

fig, ax = plt.subplots(figsize = (30,30))
ax.plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
ax.plot(qc_pred,x, label = 'filtered')
ax.set_xlabel(r'Corrected Cone Resistance $q_t$')
ax.set_ylabel(r'Depth (m)')
    
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.legend(loc='upper center',
          ncol=6)    
ax.set_title(files[7])
plt.show()



fig, ax = plt.subplots(figsize = (30,30))
ax.plot(df['Sleeve Friction fs'], df.Depth, linewidth = 1, alpha = 0.5, label = 'original')
ax.plot(fs_pred,x, label = 'filtered')
ax.set_xlabel(r'Sleeve Friction $f_s$')
ax.set_ylabel(r'Depth (m)')
    
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.legend(loc='upper center',
          ncol=6)    
ax.set_title(files[7])
plt.show()




df['Smooth qt'] = qc_pred
df['Smooth fs'] = fs_pred
f=files[7]
f = f.replace("datasets/cpt_reformatted_datasets_untouched/", "datasets/cpt_filtered_datasets/")
df.to_csv(f)