#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:57:22 2026

@author: w22043051
"""

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:01:55 2024

@author: Harry Birch

Spatial RHT Function File.
[1] Filtering function. Used for steps 3 & 4 in the Spatial RHT pipeline.
[2] Spatial RHT function. Used for steps 5 & 6 in the Spatial RHT pipeline.
"""

import os
import time
import math as m
import matplotlib
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# [1] Filtering function
def filtering(data,wlen,rmean,smr_xy,med_xy,nlev):     
    mode = "same"
    data_conv = signal.convolve(data, np.ones((1,1,rmean))/rmean, mode=mode)  
    data_spatial = (data_conv - signal.convolve(data_conv, np.ones((smr_xy,smr_xy,1)), mode=mode)/(smr_xy**2)) - (0.5*nlev/np.sqrt(rmean))
    data_spatial = data_spatial
    data_spatial[data_spatial < 0] = 0
    for n in range(data_spatial.shape[2]):
        data_spatial[:,:,n] = median_filter(data_spatial[:,:,n], med_xy)
    data_spatial = (data_spatial > 0).astype(np.uint8)
    
    # Pad spatially filtered data
    rw = (wlen-1)//2 # Radius of window 
    data_spatial_pad = np.pad(data_spatial, rw, mode='constant') 
    
    return data_spatial_pad

# [2] Full Spatial RHT function.
def spatial_rht(data_spatial_pad,wlen,smr_xy):    
    start_time = time.time()
    # Initalising RHT parameters   
    rw      = (wlen-1)//2 # Radius of window 
    nr      = (wlen)*3-2 
    dtheta     = np.sqrt(2)/(wlen-1) / 4.
    ntheta     = int(np.ceil(np.pi/dtheta))
    theta_acc  = (np.arange(ntheta) * dtheta )
    rvec       = np.arange(nr)/3. - rw   
    xacc       = (rvec[:,np.newaxis] * np.cos(theta_acc)[np.newaxis,:]).astype(int) + rw 
    yacc       = (rvec[:,np.newaxis] * np.sin(theta_acc)[np.newaxis,:]).astype(int) + rw 
    
    # Initialising matrices and variables for spatial rht
    rht_xy        = np.zeros_like(data_spatial_pad).astype(float)*np.nan
    rht_xy_reslen = np.zeros_like(data_spatial_pad).astype(float)*np.nan
    rht_xy_err    = np.zeros_like(data_spatial_pad).astype(float)*np.nan
       
    print("*** Running RHT ***")
    # Run spatial RHT code
    with tqdm(total=data_spatial_pad.shape[2]) as pbar:
        for n in range(data_spatial_pad.shape[2]):
          wpts = np.nonzero(data_spatial_pad[:,:,n]) 
          if wpts[0].size > 0:
            for pt in range(len(wpts[0])):
              xc, yc = wpts[0][pt], wpts[1][pt] 
              accum  = np.sum(data_spatial_pad[xc-rw:xc+rw+1, yc-rw:yc+rw+1, n][yacc,xacc], axis=0)
              try:
                  # RHT calculations
                  accum       = accum/np.max(accum)
                  max_acc     = np.max(accum)
                  accum_thres = accum * (accum >= (max_acc - 0.25)) 
                  theta_thres = theta_acc
                  S           = np.sum(np.sin(2*theta_thres)*accum_thres) #y
                  C           = np.sum(np.cos(2*theta_thres)*accum_thres) #x
                  mean_angle  = 0.5 * np.arctan2(S,C) 
                  ts          = np.sum(np.sin(2*theta_thres)*accum_thres)/np.sum(accum_thres) 
                  tc          = np.sum(np.cos(2*theta_thres)*accum_thres)/np.sum(accum_thres) 
                  res_len     = np.sqrt(ts**2 + tc**2) 
                  alpha2      = np.sum(accum_thres * np.cos(2*(theta_thres - mean_angle)))/np.sum(accum_thres) 
                  ng          = len(np.where(accum_thres != 0)[0]) 
                  sig2        = np.sqrt((1-alpha2)/(2*ng*res_len**2)) 
                  try:
                      err2    = np.arcsin(1.96*sig2) 
                  except:
                      err2    = np.nan
                      
                  # Store values in arrays. Filter mean_angle based on 75% of res_length
                  rht_xy[xc, yc, n] = mean_angle if res_len >= 0.75 else np.nan
                  rht_xy_reslen[xc,yc,n] = res_len 
                  rht_xy_err[xc,yc,n]    = err2 
                
              except ValueError:
                  pass
          pbar.update(1)
    pbar.close() 
    # results with padding removed. Results stored in a dictionary
    keys = ["dimension","mean angle", "error", "res length"]
    spatial_results = [rht_xy,rht_xy_err,rht_xy_reslen]
    spatial_results = [i[rw:-rw, rw:-rw, rw:-rw] for i in spatial_results]
    spatial_results = fix_padding(spatial_results, smr_xy)
    spatial_results.insert(0, "spatial")
    spatial_results = dict(zip(keys,spatial_results))

    end_time = time.time()
    print('Total time:', round((end_time - start_time)/60,2), "minutes")
    
    return spatial_results

# [3] Padding Removal
# The amount removed is based on the smoothing value
def fix_padding(array_list, smr_xy):
    for array in array_list:
        for n in range(array.shape[2]):
            arr = array[:,:,n]
            pad_width=m.ceil(smr_xy/2)
            mask = np.zeros_like(arr)
            mask[:] = np.nan
            mask[pad_width:-pad_width, pad_width:-pad_width] = 1
            array[:,:,n] = arr * mask
    return array_list


# Rotation formula to use for the angle converion array
def rotation(coords):
    if coords[1]>=0: # positive y values (top half)
        alpha = np.arctan2(coords[1],coords[0])
    else:   # negative y values (bottom half)
        alpha = 2*np.pi + np.arctan2(coords[1],coords[0])  
    return alpha


# Input submap coordinates and create conversion angle array
def angle_map(submap):
    # Obtain coordinate map from the smap
    coordinates = sunpy.map.all_coordinates_from_map(submap)
    # Create alpha array
    alpha = np.zeros_like(spatial["mean angle"][:,:,0])*np.nan  
    
    wpts = np.nonzero(coordinates)
    for pt in range(len(wpts[0])):
        xc, yc  = wpts[0][pt], wpts[1][pt]
        coords_arcsecs = coordinates[xc,yc]
        coords         = (coords_arcsecs.Tx.value,coords_arcsecs.Ty.value)
        alpha[xc,yc] = rotation(coords)
    # Plot result 
    plt.imshow(alpha*180/np.pi, origin="lower")
    plt.colorbar()
    
    return alpha


# convert polar angles
# Input rht_data to convet and conversion angle array. Returns and saves polar angle RHT cube
def convert_polar(rht_data,alpha):
    # Create polar array
    polar = np.zeros_like(rht_data)*np.nan  
    # Iterate over non-NAN values in the rht_data array
    for n in range(rht_data.shape[-1]):  
        wpts2 = np.argwhere(~np.isnan(rht_data[:,:,n]))
        for pt in range(wpts2.shape[0]):
            xc,yc = wpts2[pt][0], wpts2[pt][1]
            polar[xc,yc] = ((alpha[xc,yc]- rht_data[xc,yc]) % (np.pi))- 0.5*np.pi #- np.pi

    return polar

def classify_structure(polar_data):
    
    angles = np.asarray(polar_data)

    radial_score = np.mean(np.abs(angles) / 90)
    tangential_score = 1 - radial_score

    if tangential_score == 0:
        ratio = np.inf
    else:
        ratio = radial_score / tangential_score
    
    classification = "Radial" if ratio >= 1 else "Tangential"

    return classification, ratio
