#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 21:44:29 2022

@author: goharshoukat
This script is used to plot the location of the testing and training datasets

"""

import pandas as pd
import numpy as np
l = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
l['lat'] = l['lat'].astype(float)
l['lng'] = l['lng'].astype(float)
l['CPT'] = ['CPT_'+ x +'.csv' for x in l.CPT ]
l.loc[1, 'CPT'] = 'CPT_02.csv'
l.loc[6, 'CPT'] = 'CPT_09.csv'

test = pd.read_csv('datasets/summary.csv', usecols = ['test']).dropna()
train = pd.read_csv('datasets/summary.csv', usecols = ['train']).dropna()

l['class'] = np.nan
for i in range(len(l)):
    if l.CPT[i] in str(test.test):
        l.loc[i, 'class'] = 'test'
    elif l.CPT[i] in str(train.train):
        l.loc[i, 'class'] = 'train'
    else:
        l.loc[i, 'class'] = 'reject'
    
l_test = l[l['class'] == 'test'].reset_index(drop=True)
l_train = l[l['class'] == 'train'].reset_index(drop=True)
l_rejected = l[l['class'] == 'reject'].reset_index(drop=True)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
#ploting using cartopy and axis

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
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
def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
#rotated_crs = ccrs.RotatedPole(pole_longitude=120.0, pole_latitude=70.0)

plt.figure(figsize  = (30,30))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=True)
ocean110 = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
        scale='10m', edgecolor='none', facecolor=cartopy.feature.COLORS['water'])

ax.coastlines(resolution='10m')
ax.scatter(l_train['lng'], l_train.lat, marker='x', color='red', s = 200, zorder = 200, 
           transform= ccrs.PlateCarree(), label = 'Train')
ax.scatter(l_test['lng'], l_test.lat, marker='o', color='blue', s = 200, zorder = 200, 
           transform= ccrs.PlateCarree(), label = 'Test')
ax.scatter(l_rejected['lng'], l_rejected.lat, marker='+', color='green', s = 200, zorder = 200, 
           transform= ccrs.PlateCarree(), label = 'Reject')

ax.add_feature(ocean110)
plt.legend(loc = 'upper left')
#plt.savefig('output/plots/location_map.pdf')

ax.set_extent([-6.05, -5.85, 53.64, 53.9], ccrs.PlateCarree())
scale_bar(ax, 2)

ax.set_xticks([-6.05, -6, -5.95, -5.9, -5.85], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
ax.xaxis.set_major_formatter(lon_formatter)

ax.set_yticks([53.65, 53.7, 53.75, 53.8, 53.85, 53.9], crs=ccrs.PlateCarree())
lat_formatter = LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)


stamen_terrain = cimgt.Stamen('terrain-background')
ax.add_image(stamen_terrain, 8)
# Read image
lat = 53.67
lon = -6.025
img = Image.open('north.png')

# Plot the map

# Use `zoom` to control the size of the image#
imagebox = OffsetImage(img, zoom=0.05) 
imagebox.image.axes = ax
ab = AnnotationBbox(imagebox, [ lon, lat], pad=0, frameon=False)
ax.add_artist(ab)


#mark the location of the testing sites with alphabets
alphaMarkers= ['A', 'B', 'C', 'D']
for i in range(len(l_test)):
    plt.annotate(alphaMarkers[i], (l_test.loc[i, 'lng'] + 0.005, l_test.loc[i, 'lat'] + 0.005), 
                 fontsize = 15, zorder=200)

plt.annotate('E', (l_train.loc[0, 'lng'] + 0.005, l_train.loc[0, 'lat'] + 0.005), 
                 fontsize = 15, zorder=200)

plt.show()
plt.savefig('test_train_marker.png', dpi = 350, transparent=True)