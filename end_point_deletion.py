#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 21:19:19 2022

@author: goharshoukat
script to plot the cpts for locatios 9, 11, 23a as subplots
"""
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 14}

matplotlib.rc('font', **font)
files = sorted(glob('datasets/cpt_reformatted_datasets_untouched/*.csv'))
fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(30,30))
ax[0].set_yticks(np.linspace(0, 24, 13))

#cpt 09
df = pd.read_csv(files[7])
#for cpt09, remove data points after index 2088 and rewrite it to the same folder
df2 = df.iloc[:2089]
df2.to_csv('datasets/cpt_reformatted_datasets_with_cutoff/CPT_09.csv')
ax[0].plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 2, alpha = 0.5, label = 'Raw')
ax[0].plot(df2['Corrected Cone Resistance qt'], df2.Depth, linewidth = 2, alpha = 0.5, label = 'Adjusted')

ax[0].set_title('CPT S. No 9')
ax[0].set_xlabel(r'Corrected Cone Resistance $q_t (MPa)$')
ax[0].set_ylabel(r'Depth $(m)$')
       
   
ax[0].invert_yaxis()
ax[0].xaxis.set_label_position('top')
ax[0].xaxis.tick_top()
ax[0].grid()




#cpt 11
df = pd.read_csv(files[9])
df2 = df.iloc[:770]
df2.to_csv('datasets/cpt_reformatted_datasets_with_cutoff/CPT_11.csv')
ax[1].plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 2, alpha = 0.5, label = 'Raw')
ax[1].plot(df2['Corrected Cone Resistance qt'], df2.Depth, linewidth = 2, alpha = 0.5, label = 'Adjusted')
ax[1].set_title('CPT S. No 11')
ax[1].set_xlabel(r'Corrected Cone Resistance $q_t (MPa)$')
#ax[1].set_ylabel(r'Depth $(m)$')
       
   
#ax[1].invert_yaxis()
ax[1].xaxis.set_label_position('top')
#ax[1].xaxis.tick_top()
ax[1].grid()
ax[1].xaxis.tick_top()




#cpt 23a
df = pd.read_csv(files[17])
df2 = df.iloc[:1615]
df2.to_csv('datasets/cpt_reformatted_datasets_with_cutoff/CPT_23a.csv')
ax[2].plot(df['Corrected Cone Resistance qt'], df.Depth, linewidth = 2, alpha = 0.5, label = 'Raw')
ax[2].plot(df2['Corrected Cone Resistance qt'], df2.Depth, linewidth = 2, alpha = 0.5, label = 'Adjusted')
ax[2].set_title('CPT S. No 23a')
ax[2].set_xlabel(r'Cone Resistance $q_t (MPa)$')
#ax[1].set_ylabel(r'Depth $(m)$')
       
   
#ax[1].invert_yaxis()
ax[2].xaxis.set_label_position('top')
#ax[1].xaxis.tick_top()
ax[2].grid()
ax[2].xaxis.tick_top()

plt.legend()
plt.savefig('output/plots/9_11_23a_cutoff_plots.png', dpi = 600, transparent = True)