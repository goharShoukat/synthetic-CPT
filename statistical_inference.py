#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:54:31 2022

@author: goharshoukat

Script to plot and interpolate the different 
"""

from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from utilities import plot_cpt, cluster_plot_cpt, cluster_cpt_and_location, cpt_and_map

direc = 'datasets/cpt_raw_data/'
files = (glob(direc+ '*.csv'))
files = np.sort([x.replace('datasets/cpt_raw_data/', '') for x in files])
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

'''
cols = ['Depth', 'Cone Resistance qc', 'Sleeve Friction fs']

#prepare training dataset
train_df = pd.DataFrame()
for f in files[:3]:
    df = pd.read_csv(direc + f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols = cols)
    p_data = pd.read_csv(direc + f, encoding = 'unicode_escape', nrows = 6,
                     header=None, index_col = [0], usecols = [0,1]) # point data for lat/lng and depth
    df = df.dropna()

    df['bathymetry'] = (np.ones(len(df)) * float(p_data.loc['Water Depth']))
    df['lat'] = (np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lat']))
    df['lng'] = (np.ones(len(df)) * float(location[location['CPT'] == f[:-4]]['lng']))
    train_df = train_df.append(df)
    
df1 = train_df.iloc[:(2947-1302)]
df2 = train_df.iloc[(2947-1302):2947]
df3 = train_df.iloc[(2947):]


plt.plot(df3.Depth, df3['Cone Resistance qc'])
plt.plot(df2.Depth, df2['Cone Resistance qc'])
plt.plot(df1.Depth, df1['Cone Resistance qc'])

x = [df1.lat.iloc[0], df2.lat.iloc[0], df3.lat.iloc[0]]
y = np.linspace(pd.concat([df1.Depth, df2.Depth, df3.Depth]).min(), 
                pd.concat([df1.Depth, df2.Depth, df3.Depth]).max(), 
                1645)
X, Y = np.meshgrid(x, y)


temp1= np.empty((len(df1) - len(df2)))*np.nan
d2 = np.append(df2['Cone Resistance qc'].to_numpy(), temp1)
temp2 = np.empty(np.abs(len(df3)-len(df1))) * np.nan
d3 = np.append(df3['Cone Resistance qc'].to_numpy(), temp2)

z = (np.array([df1['Cone Resistance qc'].to_numpy(), d2, d3])).T

fig = plt.figure()
plt.contour(X, Y, z)
plt.colorbar()
'''
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
    p = pd.read_csv(direc + f, skiprows=8, encoding = 'unicode_escape',
                 skip_blank_lines=True, usecols=cols).dropna()
    p['Depth'] = np.round(p['Depth'], 2)
    p = p.rename(columns = {'Cone Resistance qc' : f[:-4]})
    dfs[f] = p


#calcuate the global mean at every height
#construct a dataframe with horizontally stacked deck by depths
dff = dfs[files[0]]
names = list(location['CPT'])
for i, f in enumerate(files[1:]):
    print(i+2, f)
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
plot_cpt(dff, col_name='CPT_01', avg_plot=True, 
        directory = 'output/plots/statistical comparison/')

cluster_plot_cpt(dff, 'output/plots/statistical comparison/')
cluster_cpt_and_location(dff, location = location, directory = 'output/plots/statistical comparison/Clusters/')
df = dff
for col in dff.columns[1:-1]: 
    cpt_and_map(dff, col, location, 'output/plots/statistical comparison/')
    
# =============================================================================
#     
# 
# Statistical Calculations here
# 
# 
# =============================================================================
