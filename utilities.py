#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:48:39 2022

@author: goharshoukat

Utilities script to save miscellaneious plotting, mapping, data reading functions

"""

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
font = {'size'   : 12}

matplotlib.rc('font', **font)

def plot_cpt(df, col_name,  directory, avg_plot : False):
    #Inputs
    #df    : dataframe : dataframe with one column = depth, data, avg of that data
    #directory : str : string for saving the figures
    #col_name : str : column name to be plotted
    #output
    #figures
    fig, ax = plt.subplots(figsize = (30,30))
    ax.plot(df[col_name], df.Depth, linewidth = 0.5, alpha = 0.5, label = col_name)
    if avg_plot:
        ax.plot(df['depth average'], df.Depth, linewidth = 0.5, alpha = 0.5, label = 'Global Average')
    
    ax.set_xlabel(r'Cone Resistance $q_c (kg/m^2)$')
    ax.set_ylabel(r'Depth $(m)$')
    x = np.arange(0, 35, 5) # define the x to make ti the same for all cpts
    y = np.arange(0, 35, 5)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    ax.legend(loc='upper center',
            ncol=6)
    plt.show()
    plt.savefig(directory + col_name + '.pdf')

def cluster_plot_cpt(df, directory):
    #Input
    #df : dataframe : column 1 will be depth colum with the remaining ones 
    #cpt prpfiles
    #last column will have row wise averages for each depth
    #directory : str : path to save the data in
    
    
    names = np.reshape(df.columns[1 :-1], (4, 6))
    fig, axs = plt.subplots(4,6, figsize = (30, 30), sharey = True)
    #axis labels
    x = np.arange(0, 50, 8) # define the x to make ti the same for all cpts
    y = np.arange(0, 30, 5)
    for i in (range(4)):
        for j in range(6):
            axs[i, j].plot(df[names[i, j]], df.Depth, linewidth = 0.7, alpha = 0.5, label = 'Site Data')
            axs[i, j].set_title(names[i, j])
            axs[i, j].plot(df['depth average'], df.Depth, linewidth = 0.7, alpha = 0.5, label = 'Global Depth Average')
            axs[i, j].set_xticks(x)
            axs[i, j].set_yticks(y)
            axs[i, j].xaxis.set_label_position('top')
            axs[i, j].xaxis.tick_top()
            axs[i, j].grid()
            if i == 0:
                axs[i, j].tick_params(labeltop = True)
            else:
                axs[i, j].tick_params(labeltop = False)
            #axs[i, j].tick_params(labeltop=False)       
    
    axs[0 , 0].invert_yaxis()#only invert once, otherwise even number of inversions cause original position
    fig.supxlabel('Cone Resistance qc')
    fig.supylabel('Depth')
    plt.tight_layout()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right')
    plt.savefig(directory + '/cluster_cpt_plot.pdf')    
    
    
def cluster_cpt_and_location(df, directory, location):
    #Inputs
    #df    : dataframe : dataframe with one column = depth, data, avg of that data
    #directory : str : string for saving the figures
    #col_name : str : column name to be plotted
    #location : array : location coordinates for the 
    #output
    #figures
    names = np.reshape(df.columns[1 :-1], (4, 6))
    fig, axs = plt.subplots(4,6, figsize = (30, 30), sharey = True)
    #axis labels
    x = np.arange(0, 50, 8) # define the x to make ti the same for all cpts
    y = np.arange(0, 30, 5)
    lat = np.reshape(np.array(location.lat), (4, 6))
    lng = np.reshape(np.array(location.lng), (4, 6))
    for i in (range(4)):
        for j in range(6):
            axs[i, j].plot(df[names[i, j]], df.Depth, linewidth = 0.7, alpha = 0.5, label = 'Site Data')
            axs[i, j].set_title(names[i, j])
            axs[i, j].plot(df['depth average'], df.Depth, linewidth = 0.7, alpha = 0.5, label = 'Global Depth Average')
            axs[i, j].set_xticks(x)
            axs[i, j].set_yticks(y)
            axs[i, j].xaxis.set_label_position('top')
            axs[i, j].xaxis.tick_top()
            axs[i, j].grid()
            if i == 0:
                axs[i, j].tick_params(labeltop = True)
            else:
                axs[i, j].tick_params(labeltop = False)
            #axs[i, j].tick_params(labeltop=False)       
            
            
            #plot the map locations
            
            axins = inset_axes(axs[i, j], width="40%", height="40%", 
                               axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                               axes_kwargs=dict(map_projection=ccrs.PlateCarree()),
                               bbox_to_anchor=(0,0, 1, 1),
                               bbox_transform=axs[i, j].transAxes)
            axins.set_extent([-6.05, -5.85, 53.65, 53.9], ccrs.PlateCarree())
            axins.coastlines(resolution="10m")
            ocean110 = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
                    scale='10m', edgecolor='none', facecolor=cartopy.feature.COLORS['water'])
            axins.add_feature(ocean110)
            axins.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=True,
                            color = 'gray')
            #axins.set_xticks([-6.1, -5.95, -5.8], crs=ccrs.PlateCarree())
            #lon_formatter = LongitudeFormatter(zero_direction_label=True)
            #axins.xaxis.set_major_formatter(lon_formatter)
            
            #axins.set_yticks([53.5, 53.6, 53.7, 53.8, 53.9, 54], crs=ccrs.PlateCarree())
            #lat_formatter = LatitudeFormatter()
            #axins.yaxis.set_major_formatter(lat_formatter)
            
            stamen_terrain = cimgt.Stamen('terrain-background')
            axins.add_image(stamen_terrain, 8)
            axins.scatter(lng, lat, marker='o', color='red', s = 2, zorder = 200, 
                       transform= ccrs.PlateCarree(), label = 'Site Locations')
            axins.scatter(lng[i, j], lat[i, j], marker = 'x', color = 'black',
                          zorder = 200, label = 'Specific Site')
    
    axs[0 , 0].invert_yaxis()#only invert once, otherwise even number of inversions cause original position
    fig.supxlabel('Cone Resistance qc')
    fig.supylabel('Depth')
    plt.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right')
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0.85, 0.075))
    #lines_labels = axs[1, 1].get_legend_handles_labels()
    #fig.legend(lines, labels, loc = 'upper right')
    plt.savefig(directory + '/cpt_cluster_and_map.pdf')
    plt.savefig(directory + '/cpt_cluster_and_map.png', dpi = 300, transparent=True)


def cpt_and_map(df, col, location, directory):
    #Inpu
    #df : dataframe : dataframe with depth, cpt data and global average
    #location : dataframe : cpt name, lat and long
    #directory : str : output directory to save it
    fig, ax = plt.subplots(figsize = (4.5,8))
    ax.plot(df[col], df.Depth, linewidth = 2, alpha = 0.8, label = col)
    ax.plot(df['depth average'], df.Depth, linewidth = 2, alpha = 0.5, label = 'Global Average')
    ax.set_xlabel(r'Cone Resistance $q_c (MPa)$')
    ax.set_ylabel(r'Depth $(m)$')
    x = np.arange(0, 16, 2) # define the x to make ti the same for all cpts
    y = np.arange(0, 28, 4)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.grid()
    ax.legend(loc='upper center',
            ncol=6)
    axins = inset_axes(ax, width="40%", height="40%", 
                       axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                       axes_kwargs=dict(map_projection=ccrs.PlateCarree()),
                       bbox_to_anchor=(0,0, 0.9, 0.9),
                       bbox_transform=ax.transAxes)
    axins.set_extent([-6.05, -5.85, 53.65, 53.9], ccrs.PlateCarree())
    axins.coastlines(resolution="10m")
    ocean110 = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
            scale='10m', edgecolor='none', facecolor=cartopy.feature.COLORS['water'])
    axins.add_feature(ocean110)
    axins.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=True,
                    color = 'gray')
    axins.set_xticks([-6.05, -5.85], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    axins.xaxis.set_major_formatter(lon_formatter)
    
    axins.set_yticks([53.7 , 53.8, 53.9], crs=ccrs.PlateCarree())
    lat_formatter = LatitudeFormatter()
    axins.yaxis.set_major_formatter(lat_formatter)
    
    stamen_terrain = cimgt.Stamen('terrain-background')
    axins.add_image(stamen_terrain, 8)
    axins.scatter(location.lng, location.lat, marker='o', color='red', s = 2, zorder = 200, 
               transform= ccrs.PlateCarree(), label = 'Site Locations')
    axins.scatter(location[location['CPT']==col]['lng'].iloc[0], 
                  location[location['CPT']==col]['lat'].iloc[0],
                  marker = 'x', color = 'black',
                  zorder = 200)
    #plt.savefig(directory + col +'_cpt_and_map.pdf')
    ax.yaxis.labelpad = 0.05
    plt.savefig(directory + col +'_cpt_and_map.png', dpi = 360, transparent=True)
    plt.close()