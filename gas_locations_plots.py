#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:53:33 2022

@author: goharshoukat

This script is used to plot the location and the cone resistance values of the
four cpt locations which were in gaseous sub-layers
"""
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from utilities import plot_cpt, cluster_plot_cpt, cluster_cpt_and_location, cpt_and_map

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import patheffects
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
import cartopy.io.img_tiles as cimgt
import numpy as np

import matplotlib
import os

font = {'size'   : 14}

direc = 'datasets/cpt_raw_data/'
files = os.listdir(direc)
files = np.sort([x.replace('datasets/cpt_reformatted_datasets_with_cutoff/', '') for x in files])[1:]
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
location['lat'] = location['lat'].astype(float)
location['lng'] = location['lng'].astype(float)
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'
del tmp

files[0][:-4]


# %% Resample at fixed intervals. 
# =============================================================================
# 
# Resampling done here 
# 
# =============================================================================
cols = ['Depth', 'Cone Resistance qc']
#read all the df into a cache
dfs = {}
for f in files:
    p = pd.read_csv(direc+f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols=cols).dropna()
    p['Depth'] = np.round(p['Depth'], 2)
    p = p.rename(columns = {'Cone Resistance qc' : f[:-4]})
    dfs[f] = p


#calcuate the global mean at every height
#construct a dataframe with horizontally stacked deck by depths
dff = dfs[files[0]]
names = list(location['CPT'])
for i, f in enumerate(files[1:]):
    #print(i+2, f)
    #determine if the merge will be left or right.     
    if len(dff) > len(dfs[f]):
        #left merge if true
        how = 'left'
    else:
        how = 'right'
        

    dff = dff.merge(
    dfs[f], on = ['Depth'],
    how = how#, suffixes=['_' + names[i+1],'_' +  names[i+2]]
        )


dff['depth average'] = np.mean(dff, axis = 1)


# =============================================================================
# 
# 
# start plotting
# 
# 
#
# =============================================================================
df = dff

gas_loc = ['CPT_03', 'CPT_05', 'CPT_05a', 'CPT_06', 'CPT_08']

for col in gas_loc: 
    cpt_and_map(dff, col, location, 'output/plots/')

    
    