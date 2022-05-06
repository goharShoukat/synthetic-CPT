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
import matplotlib.pyplot as plt
##############################################################################
direc = 'cpt_raw_data/'
files = (glob(direc+ '*.csv'))
files = np.sort([x.replace('cpt_raw_data/', '') for x in files])
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'

cols = ['Depth', 'Cone Resistance qc', 'Sleeve Friction fs']



for file, i in zip(files, range(len(files))):
    df = pd.read_csv('cpt_raw_data/' + file, skiprows=8, encoding = 'unicode_escape',
             skip_blank_lines=True, usecols = cols).dropna()
    fig, ax = plt.subplots(figsize = (30,30))
    ax.plot(df['Cone Resistance qc'], df.Depth)
    ax.set_xlabel(r'Cone Resistance $Q_c$')
    ax.set_ylabel(r'Depth (m)')
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    ax.set_title(file + '\n' + 'Lat: {}, Long: {}'.format(
        location.loc[location['CPT']==file[:-4], 'lat'].iloc[0],
        location.loc[location['CPT']==file[:-4], 'lng'].iloc[0]))
    plt.savefig(r'output/Original Dataset/' + file[:-4] + '.pdf')
    plt.close()


##############################################################################


##############################################################################
