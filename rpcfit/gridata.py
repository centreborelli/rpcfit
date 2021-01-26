#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:36:42 2020

@author: rakiki
"""
import numpy as np
import rasterio
# dataset construction
def pointCube(dim1_left, dim1_right, dim2_left, dim2_right
              , alt_min, alt_max
              , grid_len, num_layers):
    
    '''
    Construct a 3D meshgrid 
    
    Args:
    dim1, dim2 can either be lon, lat (or lat, lon /order is irrelevant)
                            or row, col (or col, row /order is irrelevant )
    dim1/2_left/right: the limits of the dimension range
    alt_min/max: the limits of the altitude range
    grid_len: the number of points to be taken in dim1/2
    num_layers: the number of points to be taken in the altitude range
    Returns: 
    dim1_list, dim2_list, alt_list: a 1D numpy array of values in each dimension
    '''
    # meshgrid 
    dim1_rg = np.linspace(dim1_left,dim1_right, grid_len)
    dim2_rg = np.linspace(dim2_left, dim2_right, grid_len)
    alt_rg = np.linspace(alt_min,alt_max, num_layers)
    dim1_grd, dim2_grd, alt_grd = np.meshgrid(dim1_rg, dim2_rg, alt_rg)
    dim2_list = dim2_grd.ravel()
    dim1_list = dim1_grd.ravel()
    alt_list = alt_grd.ravel()
    return dim1_list, dim2_list, alt_list

def read_dem(dem_path):
    '''
    Reads the dem that covers the area of intererst, 
    geotiff, from the disk
    dem_path: the path to the dem file
    Returns:
        demdb: rasterio closed dataset
        demdata: numpy.2D array containing the dem 
    '''
    with rasterio.open(dem_path) as demdb:
        demdata = demdb.read(1)
    return demdb, demdata

def getDataset_projection(demdb, demdata , grid_len, num_layers, projectionFunc
                          , train = True, margin = 0.2, **kwargs):
    '''
    Computes 3D pts + 2D correspondence for train or test set
    Args: 
    demdb: rasterio db of the geotiff dem on the area of interest
    demdata: the data of the dem 
    grid_len: the len of the grid in the two lon, lat dimensions 
    num_layers: the number of alt layers
    projectionFunc: projection function (lon, lat, alt) -> (col, row)
    train: if True, returns a training grid else returns a test grid (shifted by half a step from the train grid)
    margin: the safety margin to apply to the altitude bounds when constructing the grid
    kwargs: dict of params to pass to projection function 
    Returns: 
    input_locs: [lon, lat, alt] array
    target: [col, row] array
    '''
    lon_left, lat_top = demdb.transform * (0,0)
    lon_right,lat_bottom =  demdb.transform * (demdb.width -1 ,demdb.height -1 )
    mask = (demdata != demdb.nodata)
    alt_min = np.nanmin(demdata[mask])
    alt_max = np.nanmax(demdata[mask])
    alt_margin = np.round((alt_max - alt_min) * margin)
    alt_min -= alt_margin
    alt_max += alt_margin
    if train:
        lon,lat, alt = pointCube(lon_left, lon_right, lat_top, 
                                           lat_bottom, alt_min , 
                                           alt_max, grid_len, num_layers, 
                                           )
    else:
        lon_stp  = (lon_right - lon_left)/(2 * (grid_len - 1 ) ) 
        lat_stp = (lat_bottom - lat_top)/(2 * (grid_len - 1 ) ) 
        alt_stp = np.round((alt_max - alt_min)/(2 * (num_layers - 1 ) ) )  
        lon, lat, alt = pointCube(lon_left + lon_stp , lon_right + lon_stp, lat_top + lat_stp, 
                                                lat_bottom + lat_stp, alt_min + alt_stp, alt_max + alt_stp
                                                , grid_len, num_layers, 
                                               )
    col, row = projectionFunc( lon = lon , lat = lat
                              , alt = alt , **kwargs)
    input_locs = np.vstack((lon, lat, alt)).T
    target =  np.vstack((col,row)).T
    return input_locs, target

def getDataset_localization(demdb, demdata, grid_len, num_layers
                            , im_size , localizationFunc, train = True, margin = 0.2 
                            , **kwargs ):
    '''
    Computes 3D pts + 2D correspondence for train or test set
    
    Args: 
    demdb: rasterio db of the geotiff dem on the area of interest
    demdata: the data of the dem
    grid_len: the len of the grid in the two lon, lat dimensions 
    num_layers: the number of alt layers
    im_size: tuple(height, width) of the image
    localizationFunc: localization function (col, line, alt) -> (lon, lat)
    train: if True, returns a training grid else returns a test grid (shifted by half a step from the train grid)
    margin: the safety margin to apply to the bounds of the image dimension and the altitude bounds when constructing the grid
    kwargs: localization function additional arguments
    Returns: 
    input_locs: [lon, lat, alt] array
    target: [col, row] array
    '''
    # line, col limits
    lines = im_size[0]
    l_margin = np.round(margin * lines)
    columns = im_size[1]
    c_margin = np.round(margin * columns)

    # alt limits, use preexisting demdb, demdata
    mask = (demdata != demdb.nodata)
    alt_min = np.nanmin(demdata[mask]) 
    alt_max = np.nanmax(demdata[mask])
    alt_margin = np.round((alt_max - alt_min) * margin)

    if train:
        c,l, alt = pointCube(-c_margin, columns + c_margin, -l_margin, 
                            lines + l_margin, alt_min - alt_margin , alt_max + alt_margin,
                             grid_len, num_layers)
    else:
        c_stp  = (columns + 2 * c_margin)/(2 * (grid_len - 1 ) ) 
        l_stp = (lines + 2 * l_margin)/(2 * (grid_len - 1 ) ) 
        alt_stp = np.round((alt_max - alt_min + 2 * alt_margin)/(2 * (num_layers - 1 ) ) )  
        c, l, alt = pointCube(-c_margin + c_stp , columns + c_margin + c_stp, -l_margin + l_stp, 
                                                lines + l_margin + l_stp, alt_min - alt_margin + alt_stp
                                                  , alt_max + alt_margin + alt_stp, 
                                                grid_len, num_layers)
    lon , lat = localizationFunc(col = c, line = l, alt = alt, **kwargs)
    input_locs = np.vstack((lon, lat, alt)).T
    target =  np.vstack((c,l)).T
    return input_locs, target