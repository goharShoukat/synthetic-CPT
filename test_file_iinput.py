#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:17:06 2022

@author: goharshoukat

Creat test input files to test the models

adds a column of latitude, longitude and bathymetry
"""

import pandas as pd
import numpy as np
from glob import glob

direc = 'datasets/cpt_raw_data/'
files = (glob(direc+ '*.csv'))
files = np.sort([x.replace('datasets/cpt_raw_data/', '') for x in files])
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'

cols = ['Depth', 'Cone Resistance qc', 'Sleeve Friction fs']
outdir = 'datasets/cpt_reformatted_datasets/'

for file in files:
    df = pd.read_csv('datasets/cpt_raw_data/' + file, skiprows=8,
        encoding = 'unicode_escape', skip_blank_lines=True,
        usecols = cols).dropna()

    df['latitude'] = pd.Series(np.ones(len(df)) *
                   location[location['CPT']==file[:-4]]['lat'].iloc[0])

    df['longitude'] = pd.Series(np.ones(len(df)) *
                float(location[location['CPT']==file[:-4]]['lng'].iloc[0]))

    df.to_csv(outdir + file, index = False)
