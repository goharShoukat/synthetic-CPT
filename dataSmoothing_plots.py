#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 21:38:40 2022

@author: goharshoukat

This script is used to plot the cpt standard profile and the smoothed out profile

"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import peakutils
import numpy as np
import matplotlib as mpl
import string
mpl.rcParams['font.size'] = 12
labels = list(string.ascii_uppercase)[:4]

files_fil = sorted(os.listdir('datasets/cpt_reformatted_datasets_untouched/'))[1:]
files_raw = np.delete(np.array(sorted(os.listdir('datasets/cpt_reformatted_datasets_untouched/'))), 0)

fil=pd.read_csv('datasets/cpt_filtered_datasets/' + files_fil[0], engine = 'python')

raw=pd.read_csv('datasets/cpt_reformatted_datasets_untouched/' + files_raw[0], engine = 'c')
peaks = peakutils.indexes(raw['Corrected Cone Resistance qt'], thres=0.2, min_dist=10)


fig, ax = plt.subplots(figsize = (4.5,10))
ax.plot(raw['Corrected Cone Resistance qt'], raw.Depth, linewidth = 1, alpha = 0.5, label = 'Raw')
ax.plot(fil['Smooth qt'], fil.Depth, label = 'Smoothed')
#ax.plot(raw['Corrected Cone Resistance qt'][peaks], raw.Depth[peaks], marker = 'o', ls = "")

ax.set_xlabel(r'Corrected Cone Resistance $q_t (MPa)$')
ax.set_ylabel(r'Depth $(m)$')
    
ax.invert_yaxis()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.grid()
ax.legend(loc='upper center',
          ncol=6)    
#ax.set_title(f)
ax.yaxis.set_label_coords(-0.08,0.5)
xpoints = raw['Corrected Cone Resistance qt'][peaks]
ypoints =  raw.Depth[peaks]
for label, x, y in zip(labels, xpoints, ypoints):
    plt.annotate(
      label,
      xy=(x, y), xytext=(20, 10),
      textcoords='offset points', ha='center', va='bottom',
      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()

plt.savefig('output/plots/smoothing.png', dpi = 600, transparent=True)
