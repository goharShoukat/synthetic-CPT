#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:22:32 2022

@author: goharshoukat

Script to plot the location of the CPT profiles
"""
import pandas as pd
import numpy as np
l = pd.read_csv('location.csv', usecols=['CPT', 'lat', 'lng'])
l['lat'] = l['lat'].astype(float)
l['lng'] = l['lng'].astype(float)
l['CPT'] = ['CPT_'+ x +'.csv' for x in l.CPT ]
test = pd.read_csv('datasets/summary.csv', usecols = ['test']).dropna()
train = pd.read_csv('datasets/summary.csv', usecols = ['train']).dropna()

l['class'] = np.nan
for i in range(len(l)):
    if l.CPT[i] in str(test.test):
        l.loc[i, 'class'] = 'test'
    else:
        l.loc[i, 'class'] = 'train'
    

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
#ploting using cartopy and axis

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
ocean110 = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
        scale='10m', edgecolor='none', facecolor=cartopy.feature.COLORS['water'])
ax.set_extent([-7, -4, 52, 54.5], ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
ax.scatter(l['lng'], l.lat, marker='o', color='red', s = 10, zorder = 200, 
           transform= ccrs.PlateCarree(), label = 'Site Locations')
ax.add_feature(ocean110)
plt.legend()
#plt.savefig('output/plots/location_map.pdf')

axins = inset_axes(ax, width="40%", height="40%", 
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=ccrs.PlateCarree()),
                   bbox_to_anchor=(0.5,-0.5, 1, 1), bbox_transform=ax.transAxes)
axins.set_extent([-6.05, -5.85, 53.64, 53.9], ccrs.PlateCarree())
axins.scatter(l['lng'], l.lat, marker='o', color='red', s = 20, zorder = 200, 
           transform= ccrs.PlateCarree())
axins.coastlines(resolution="10m")
axins.add_feature(ocean110)
axins.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=True)
ax.indicate_inset_zoom(axins, edgecolor="black")
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls = '--', lw = 0.5)
scale_bar(ax, 20)
scale_bar(axins, 2)
axins.set_xticks([-6.05, -5.85], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axins.xaxis.set_major_formatter(lon_formatter)

axins.set_yticks([53.64, 53.9], crs=ccrs.PlateCarree())
lat_formatter = LatitudeFormatter()
axins.yaxis.set_major_formatter(lat_formatter)

stamen_terrain = cimgt.Stamen('terrain-background')
ax.add_image(stamen_terrain, 8)
axins.add_image(stamen_terrain, 8)
# Read image
lat = 52.1
lon = -6.75
img = Image.open('north.png')

# Plot the map

# Use `zoom` to control the size of the image
imagebox = OffsetImage(img, zoom=0.05) 
imagebox.image.axes = ax
ab = AnnotationBbox(imagebox, [ lon, lat], pad=0, frameon=False)
ax.add_artist(ab)
plt.show()
plt.savefig('map2.pdf', transparent=True)
plt.savefig('map2.png', dpi = 300, transparent=True)