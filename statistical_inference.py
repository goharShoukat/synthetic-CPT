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
from matplotlib.cm import ScalarMappable
#from pylab import *
from scipy.interpolate import griddata
from scipy import interpolate
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
# Statistical interpolation
# 
# 
# =============================================================================
#dfs['CPT_01.csv']['lat'] = location[location['CPT']=='CPT_01']['lat'].iloc[0]
#dfs['CPT_01.csv']['lng'] = location[location['CPT']=='CPT_01']['lng'].iloc[0]
df1 = dff[['Depth', 'CPT_01']]
df1['lat'] = location[location['CPT']=='CPT_01']['lat'].iloc[0]
df1['lng'] = location[location['CPT']=='CPT_01']['lng'].iloc[0]


df2 = dff[['Depth', 'CPT_02']]
df2['lat'] = location[location['CPT']=='CPT_02']['lat'].iloc[0]
df2['lng'] = location[location['CPT']=='CPT_02']['lng'].iloc[0]

x = np.array(df1.Depth)
y = [df1.lat.iloc[0], df2.lat.iloc[0]]
X, Y = np.meshgrid(x, y)
z = np.array((df1.CPT_01, df2.CPT_02))



#matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z)
ax.set_xlabel('Depth')
ax.set_ylabel('Latitude')
ax.set_zlabel('qc')
ax.set_title('CPT01 and CPT02 qc linear interpolation')
plt.contourf(X, Y, z, zdir='z', alpha=1)
plt.colorbar()
# =============================================================================
# contouring at each depth level 
# 
# =============================================================================
x =  [df1.lat.iloc[0], df2.lat.iloc[0]] #latitude
y = [df1.lng.iloc[0], df2.lng.iloc[0]]
X, Y = np.meshgrid(x, y)
Z = np.ones((2,2)) * np.nan
Z[0, 0] = 0.041
Z[1, 1] = 0.0642
Z[0, 1] = 0.0525
Z[1, 0] = 0.0525
plt.contourf(X, Y, Z, zdir='z', alpha=1)
plt.colorbar()
# =============================================================================
# 
# =============================================================================
x = np.array(location.lat)
y = np.array(location.lng)
lat, lng = np.meshgrid(x, y)
Z = np.ones((24, 24)) * np.nan

lat_search = np.where(x == location.lat.iloc[0])
lng_search = np.where(y == location.lng.iloc[0])
Z[lat_search, lng_search] = dff[dff['Depth']==0.01]['CPT_01'].iloc[0]

for i, col in zip(range(24), dff.columns[1:-1]):
    lat_search = np.where(x==location.lat.iloc[i])
    lng_search = np.where(y==location.lng.iloc[i])
    Z[lat_search, lng_search] = dff[dff['Depth'] == 0.01][col].iloc[0]


plt.scatter(lng, lat, c=Z, alpha=1, s=80)
plt.xlabel(r'Longitude $^\circ$N')
plt.ylabel(r'Latitude $^\circ$E')
plt.grid()
clb = plt.colorbar()
clb.set_label('Cone Resistance qc', rotation = 90)
plt.title('Cone Resistance at Depth = 0.01cm')


fun = interpolate.interp2d(x, y, np.array(dff.loc[0][1:-1]))
latnew = np.arange(np.min(x), np.max(x), 0.01)
lngnew = np.arange(np.min(y), np.max(y), 0.01)
Znew = fun(latnew, lngnew)
plt.tricontour(x, y, np.array(dff.loc[0][1:-1]))
