#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:15:22 2022

@author: goharshoukat

scrip to plot specific cpt profiles from the best results identified after
hyperparamter tuning. different form the script plots
"""
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
location = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
location['lat'] = location['lat'].astype(float)
location['lng'] = location['lng'].astype(float)
tmp = ['0' + l for l in location.loc[:10, 'CPT'] if float(l) < 10]
for i in range(len(tmp)):
    location.loc[i , 'CPT'] = tmp[i]
location['CPT'] = (['CPT_' + location for location in location.CPT])
location.loc[21, 'CPT'] = 'CPT_05a'
del tmp
location.CPT = location.CPT + '.csv'


plt.rcParams.update({'font.size': 17})
##############################################################################
reconst_model_test_dir = os.listdir('output/Model Evaluation/Tenth Attempt/test')
if '.DS_Store' in reconst_model_test_dir:
    reconst_model_test_dir.remove('.DS_Store')
# %% read output from modelled data
#and remove the unnecessary folders in the directory.
#Tenth reconstruct results in test directory and then in the training dir
reconst_model_test_dir = os.listdir('output/Model Evaluation/Tenth Attempt/test')
if '.DS_Store' in reconst_model_test_dir:
   reconst_model_test_dir.remove('.DS_Store')


test_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_test_dir:
    files = os.listdir(r'output/Model Evaluation/Tenth Attempt/test/' + path)

    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Tenth Attempt/test/' + path):
        os.mkdir(r'output/Model Evaluation/Tenth Attempt/test/' + path)

    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:
        df = pd.read_csv(r'output/Model Evaluation/Tenth Attempt/test/' + path + '/'+ file)
        reconstructed[path][file] = df

#select only models 3 and 4. the rest produce poor results 
remove = (['Model1_opt_ADAM_activation_LeakyReLU', 'Model2_opt_ADAM_activation_LeakyReLU',
                   'Model5_opt_ADAM_activation_LeakyReLU', 'Model4_opt_ADAM_activation_LeakyReLU'])
reconstructedM1 = {key: reconstructed[key] for key in reconstructed if key not in remove}
###############################################################################
reconst_model_test_dir = os.listdir('output/Model Evaluation/Nineth Attempt/test')
if '.DS_Store' in reconst_model_test_dir:
   reconst_model_test_dir.remove('.DS_Store')


test_files = np.sort(pd.read_csv('datasets/summary.csv', usecols=['test']).dropna()).astype(str)
reconstructed = {}
for path in reconst_model_test_dir:
    files = os.listdir(r'output/Model Evaluation/Nineth Attempt/test/' + path)

    #create directory for the model output graphs for comparison
    if not os.path.isdir(r'output/Model Evaluation/Nineth Attempt/test/' + path):
        os.mkdir(r'output/Model Evaluation/Nineth Attempt/test/' + path)

    reconstructed[path] = {} #created nested multi-layered dicts
    for file in files:
        df = pd.read_csv(r'output/Model Evaluation/Nineth Attempt/test/' + path + '/'+ file)
        reconstructed[path][file] = df

#select only models 3 and 4. the rest produce poor results 
remove = (['Model1_opt_ADAM_activation_LeakyReLU', 'Model2_opt_ADAM_activation_LeakyReLU',
                   'Model3_opt_ADAM_activation_LeakyReLU', 'Model5_opt_ADAM_activation_LeakyReLU'])
reconstructedM2 = {key: reconstructed[key] for key in reconstructed if key not in remove}

m1m2 = reconstructedM1 | reconstructedM2









orig_direc = 'datasets/cpt_filtered_datasets/'
orig_files = (glob(orig_direc+ '*.csv'))
orig_files = np.sort([x.replace('datasets/cpt_filtered_datasets/', '') for x in orig_files])
depth = ['M1', 'M2']
f,(ax) = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,1]}, figsize=(30,30),
                      sharey = True)
for test_file, i in zip(test_files, range(4)):
    print(test_file[0])
    #load reformatted cpt data files
    #if str(orig_file) in str(test_files):
    df_orig = pd.read_csv('datasets/cpt_filtered_datasets/' + test_file[0])

    #compare each original datafile witht he data generated from the
    #several models. a for loop to run through each of the models
    #create one plot. for each plot, plot the 5 models and compare results

    for model, d in zip(m1m2, depth):
        #run a for loop to compare each of the test data files reconstructed
        #using the models

        df_rec = reconstructed[model]['reconstructed_' + test_file[0]]
        ax[i].plot(df_rec['fs'], df_rec.depth, label = d)

    #ax[i].plot(df_orig['Cone Resistance qc'], df_orig.Depth, label = 'Original')
    ax[i].plot(df_orig[r'Sleeve Friction fs'], df_orig.Depth, label = 'Original')
    ax[i].set_xlabel(r'$f_s (kg/m^2)$')
    title = test_file[0][:-4].replace('CPT_', 'Site ')
    ax[i].set_title('{}'.format(title))
    ax[i].xaxis.set_label_position('top')
    ax[i].grid()
    ax[i].set_xticks([0,25,50,75,100, 125])
    ax[i].xaxis.tick_top()
ax[0].invert_yaxis()
ax[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

ax[0].set_ylabel(r'Depth $(m)$')
plt.legend(loc = "lower right")
for test_file, i in zip(test_files, range(4)):
    
    axins = inset_axes(ax[i], width="40%", height="40%", 
                       axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                       axes_kwargs=dict(map_projection=ccrs.PlateCarree()),
                       bbox_to_anchor=(0,0, 0.95, 0.95),
                       bbox_transform=ax[i].transAxes)
    axins.set_extent([-6.05, -5.85, 53.65, 53.9], ccrs.PlateCarree())
    axins.coastlines(resolution="10m")
    ocean110 = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
            scale='10m', edgecolor='none', facecolor=cartopy.feature.COLORS['water'])
    axins.add_feature(ocean110)
    axins.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=True,
                    color = 'gray')
    axins.set_xticks([], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    axins.xaxis.set_major_formatter(lon_formatter)
    
    axins.set_yticks([], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    axins.yaxis.set_major_formatter(lat_formatter)
    
    stamen_terrain = cimgt.Stamen('terrain-background')
    axins.add_image(stamen_terrain, 8)
    axins.scatter(location.lng, location.lat, marker='o', color='red', s = 2, zorder = 200, 
               transform= ccrs.PlateCarree())
    axins.scatter(location[location['CPT']==test_file[0]]['lng'].iloc[0], 
                  location[location['CPT']==test_file[0]]['lat'].iloc[0],
                  marker = 'x', color = 'black',
                  zorder = 200)

    #plt.savefig(directory + col +'_cpt_and_map.pdf')
    #ax.yaxis.labelpad = 0.05
    #ax[i].set_xlim([0,3])

plt.savefig('output/plots/reconstruction.png', dpi = 500, transparent=True)

