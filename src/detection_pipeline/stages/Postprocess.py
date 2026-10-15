#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:28:15 2026

@author: Harry Birch


Takes FITs input and label file to calculate relavent parameters for prominences/AR database
"""


import os
from pathlib import Path
import sunpy.map
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
from sunpy.map.maputils import all_coordinates_from_map ,coordinate_is_on_solar_disk, solar_angular_radius 
from sunpy.coordinates import SphericalScreen, HeliographicStonyhurst, get_body_heliographic_stonyhurst
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_pointing_table, get_correction_table
from dataclasses import dataclass, field
from typing import List
import csv
from tqdm import tqdm
from astropy.stats import median_absolute_deviation
from skimage.filters import threshold_local
from scipy.ndimage import map_coordinates
from utils.Spatial_RHT_Functions import spatial_rht, filtering, convert_polar, classify_structure


@dataclass
class LabelledSubmap:
    submap: sunpy.map.Map
    centre_coord: object # Skycoord (Helioprojective)
    label_class: int()
    box_w_pix: float
    box_h_pix: float
    x_centre_pix : float
    y_centre_pix : float
    confidence: float

# =============================================================================
# Converts yolo labels to pixel coordinates
# =============================================================================
def yolo_to_pixel(labels, img_width, img_height):
    x_centre = labels[:, 1] * img_width
    y_centre = img_height - (labels[:, 2] * img_height) # flip y
    box_w    = labels[:, 3] * img_width
    box_h    = labels[:, 4] * img_height
    return x_centre, y_centre, box_w, box_h

# =============================================================================
# Calibration function 
# =============================================================================
# Calibration function for AIA data
def aia_calibrated_map(smap, pointing_table, correction_table):
    from aiapy.calibrate import register, update_pointing, correct_degradation
    
    # Update Pointing using provided pointing table
    m_updated_pointing = update_pointing(smap, pointing_table=pointing_table)
    
    # Register the image (solar disk alignment)
    m_registered = register(m_updated_pointing)
    
    # Normalise for exposure time
    m_normalized = m_registered / m_registered.exposure_time

    # Correct Degradation
    smap_corrected = correct_degradation(m_normalized, correction_table=correction_table)
    
    # Return corrected map
    smap = smap_corrected
    return smap

# =============================================================================
# Function to mask out spicules for prominence class submaps
# =============================================================================
def mask_spicules(coordinates, threshold_pix, smap):
    
    # Try to get CDELT2, default to 0.6 arcsec/pixel if missing
    plate_scale = float(f"{smap.fits_header.get('CDELT2', 0.6):.3g}")     
    # Convert threshold pixels to arcseconds using plate_scale
    threshold_arcsec = threshold_pix * plate_scale 
    
    # compute angular separation from disk center.
    r = np.arccos(np.cos(coordinates.Tx) * np.cos(coordinates.Ty))
    # returns the apparent radius of the Sun at the observation time/location.
    solar_r = solar_angular_radius(coordinates)

    return r <= (solar_r + threshold_arcsec * u.arcsec) # Mask for values above the spicule level    

# =============================================================================
# Disk removal
# =============================================================================
def remove_disk(smap, threshold_pix=None):#, mask, data=None, size=None):             
        hpc_coords = all_coordinates_from_map(smap) 
        
        if threshold_pix is not None:
            mask = mask_spicules(hpc_coords, threshold_pix, smap)
        else:
            mask = coordinate_is_on_solar_disk(hpc_coords) 
        data = smap.data
        data[mask] = np.nan  # Set outside-disk pixels to NaN
        smap = sunpy.map.Map(data, smap.meta)  
        return smap 

    
# =============================================================================
# Function to obtain the box coords of a given label array and extract submaps
# =============================================================================
def label_submaps(labels: np.ndarray, smap: sunpy.map.Map) -> List[LabelledSubmap]:
    # extract wcs
    wcs = smap.wcs
    
    img_w = smap.fits_header['NAXIS1']
    img_h = smap.fits_header['NAXIS2']
        
    # obtain centre pixel and box coords
    x_centre, y_centre, box_w, box_h = yolo_to_pixel(labels , img_w, img_h)
    
    # Get unique classes present in this label file
    classes = set(labels[:, 0].astype(int))

    smap_disk       = remove_disk(smap)                    if 1 in classes else None
    smap_prominence = remove_disk(smap, threshold_pix=20)  if 0 in classes else None

    results = []
    for i in range(len(labels)):
        # Select pre-computed map by class
        masked_smap = smap_prominence if labels[i, 0] == 0 else smap_disk
        
        # convert box corners to wcs
        x0 = x_centre[i] - box_w[i] / 2
        y0 = y_centre[i] - box_h[i] / 2
        x1 = x_centre[i] + box_w[i] / 2
        y1 = y_centre[i] + box_h[i] / 2
        
        bottom_left  = wcs.pixel_to_world(x0 ,y0) #SkyCoord(x0*u.arcsec,y0*u.arcsec, frame=smap.coordinate_frame)
        top_right    = wcs.pixel_to_world(x1, y1) #SkyCoord(x1*u.arcsec, y1*u.arcsec, frame=smap.coordinate_frame)
        centre_coord = wcs.pixel_to_world(x_centre[i],y_centre[i])
        
        results.append(LabelledSubmap(
            submap       = smap.submap(bottom_left, top_right=top_right),
            centre_coord = centre_coord,
            label_class  = int(labels[i,0]),
            box_w_pix    = box_w[i],
            box_h_pix    = box_h[i],
            x_centre_pix = x_centre[i],
            y_centre_pix = y_centre[i],
            confidence   = labels[:,-1][i]
            ))

    return results

# =============================================================================
# Use plate scale to get Mm per pixel for prominence area calculation
# =============================================================================
def get_Mm_per_pixel(smap):
    plate_scale   = smap.fits_header['CDELT2']
    rsun_ref      = smap.fits_header['RSUN_REF']
    rsun_obs      = smap.fits_header['RSUN_OBS']
    return plate_scale * (rsun_ref / rsun_obs) / 1e6

# =============================================================================
# Function to mask prominence and calcaulte area based on mean_abosolute_deviation
# =============================================================================
def mask_area_MAD_adaptive(smap, Mm_per_pixel, n_sigma=1, block_size=101, offset=0):
    # Applied MAD thresholding and then adaptive thresholding on the resulting mask
    arr = smap.data

    # --- MAD threshold ---
    background = np.nanmedian(arr)
    noise = median_absolute_deviation(arr, ignore_nan=True)
    mad_thresh = background + n_sigma * noise

    # Apply MAD threshold first
    mad_masked_arr = np.where(arr > mad_thresh, arr, np.nan)

    # --- Adaptive thresholding on MAD-masked image ---
    arr_filled = np.nan_to_num(mad_masked_arr, nan=background)

    local_thresh = threshold_local(
        arr_filled,
        block_size=block_size,
        method='gaussian',
        offset=offset
    )

    adaptive_mask = arr_filled > local_thresh

    area = np.sum(adaptive_mask) * Mm_per_pixel**2

    return adaptive_mask, area


# =============================================================================
# Calculate centre of mass based on the area mask
# =============================================================================
def centre_of_mass(smap, mask):
    # Masked array -> applied to original data
    masked_arr = np.where(mask, smap.data, np.nan)
    Y, X = np.indices(mask.shape)

    # calculate moment based on X and Y indices * pixel intensity / sum(array)
    x_c = np.nansum(X * masked_arr)/np.nansum(masked_arr)
    y_c = np.nansum(Y * masked_arr)/np.nansum(masked_arr)
    
    coord = smap.pixel_to_world(x_c * u.pix, y_c * u.pix)

    return coord

# =============================================================================
# Polar Re-projection for Height & Width
# =============================================================================
def radial_transform_submap(smap, mask, Mm_per_pixel):
    
    # Get data shape
    ny,nx = mask.shape

    # --- Computing scale to define grid ---
    ref_pix = smap.reference_pixel         # in pixels
    ref_coord = smap.reference_coordinate  # in arcsecs (Tx, Ty)

    # Pixel scale (arcsecs per pixel)
    scale_x = smap.scale.axis1.value
    scale_y = smap.scale.axis1.value

    # Compute solar centre location
    sun_cx = ref_pix.x.value - (ref_coord.Tx.value) / scale_x # pixels | equivalent to smap.fits_header["CRPIX1"] but this is shifted by 1 pixel
    sun_cy = ref_pix.y.value - (ref_coord.Ty.value) / scale_y # pixels | equivalent to smap.fits_header["CRPIX2"] but this is shifted by 1 pixel

    # define radial range
    # Distance from sun centre to ALL four corners
    corner_dists = np.sqrt(
        (np.array([0, nx, 0, nx]) - sun_cx)**2 +
        (np.array([0, 0, ny, ny]) - sun_cy)**2
        )
    r_min = corner_dists.min()
    r_max = corner_dists.max()

    # determine angular coverage
    corners_x = np.array([0, nx, 0, nx]) - sun_cx
    corners_y = np.array([0, 0, ny, ny]) - sun_cy

    # convert corners to angles
    angles = np.arctan2(corners_y,corners_x) # in radians
    # Detect wrap: if range > pi, angles have wrapped around +/-pi boundary
    if angles.max() - angles.min() > np.pi:
        # Shift negative angles up by 2*pi so they are contiguous with positive ones
        angles = np.where(angles < 0, angles + 2*np.pi, angles)

    theta_min, theta_max = angles.min(), angles.max() # angular span of the submap

    # define output polar grid
    n_r     = ny
    n_theta = nx

    # create coordinate arrays
    r = np.linspace(r_min, r_max, n_r)
    theta = np.linspace(theta_min, theta_max, n_theta)

    # Buid 2D polar coordinate grid
    # 2D polar grid → Cartesian image coords
    R, THETA = np.meshgrid(r, theta, indexing='ij')

    # Convert polar coords back to image coords
    X = R * np.cos(THETA) + sun_cx
    Y = R * np.sin(THETA) + sun_cy

    # Interpolation
    polar_image = map_coordinates(mask, [Y,X], order=1, mode='constant', cval=0.0)

    # --- Calculating Height and Width ---
    # Find width and height
    ys, xs = np.where(polar_image > 0)
    height = (r[ys].max() - r[ys].min())*Mm_per_pixel
    
    # use arc length for width     
    # radius from solar centre to midpoint in pixels
    r_characteristic = ((r[ys].max() + r[ys].min()) / 2) * Mm_per_pixel 
    width = theta[xs].max() - theta[xs].min()
    width = width * r_characteristic

    # return alpha - polar angle of each pixel relative to the Sun's center.
    # Build pixel coordinate grids
    PX, PY = np.meshgrid(np.arange(nx), np.arange(ny))   # shape (ny, nx)
    alpha = np.arctan2(PY - sun_cy, PX - sun_cx)  # shape (ny, nx), radians

    return height, width, alpha 

# =============================================================================
# Calculate submap Properties
# =============================================================================
def calculate_box_properties(labelled_submaps: List[LabelledSubmap], n_sigma=1):
    properties = []
    
    # Computed once from the first submap — same FITS file for all
    Mm_per_pixel = get_Mm_per_pixel(labelled_submaps[0].submap)

    for ls in labelled_submaps:
        
        # Extract centre coords in arcseconds
        Tx = ls.centre_coord.Tx.to(u.arcsec).value
        Ty = ls.centre_coord.Ty.to(u.arcsec).value
        
        # Calculate area, centre of mass, height, and width
        intensity_mask, area = mask_area_MAD_adaptive(ls.submap, Mm_per_pixel)
        com_coord = centre_of_mass(ls.submap, intensity_mask)
        com_Tx = com_coord.Tx.value
        com_Ty = com_coord.Ty.value
        height, width, alpha = radial_transform_submap(ls.submap, intensity_mask, Mm_per_pixel)

        # Get intensity values
        data = np.where(intensity_mask, ls.submap.data, np.nan) # mask the origina submap array
        mean_intensity = np.nanmean(data)
        max_intensity  = np.nanmax(data)
        total_flux     = np.nansum(data)                     
    
        # Spatial RHT for prominences
        if ls.label_class == 0:
            data = np.expand_dims(data, axis=2) # expand dimensions for spatial RHT
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) #  Convert nan to number
    
            # Parameters and filepaths
            smr_xy     = 30
            med_xy     = 3
            wlen       = 51
            rmean      = 1
            background = ls.submap.data[~intensity_mask]
            nlev       = np.nanmedian(np.abs(background - np.nanmedian(background)))# * 1.4826  # MAD → sigma equivalent
    
            # Run Spatial RHT
            data_spatial_pad = filtering(data, wlen, rmean, smr_xy, med_xy, nlev)
            spatial_rht_data = spatial_rht(data_spatial_pad, wlen, smr_xy)
            # filtered mean angle based on 75 % of res length
            mean_angle = spatial_rht_data["mean angle"] # filter redundant - arleady done in Spatial RHT funciton
            # mean_angle = np.where(spatial_rht_data["res length"]>=0.75,
            #                       spatial_rht_data["mean angle"], np.nan)
            
            # RHT coverage
            bool_mask = intensity_mask.astype(bool)
            rht_mask  =  ~np.isnan(mean_angle[:,:,0])
            rht_coverage = np.nansum(np.logical_and(bool_mask,rht_mask))/np.nansum(bool_mask)

            # Convert to Spatial RHT result to polar
            polar = convert_polar(mean_angle, alpha)*180/np.pi
            # mask out limb contribution
            smap_polar = sunpy.map.Map(polar[:,:,0], ls.submap.meta)
            polar = remove_disk(smap_polar, threshold_pix=30)
            
            # Extract non-nan pixels and classify
            polar_data = polar.data[~np.isnan(polar.data)]    
            _,ratio = classify_structure(polar_data)
        else:
            ratio, rht_coverage = None, None

        if np.isnan(Tx) or np.isnan(Ty):
            print(f"Warning: NaN centre_coord for class {ls.label_class} box, latitude set to None.")
            latitude = None
        else:
            # Observer coordinates from smap
            observer = ls.submap.observer_coordinate
            # Use SphericalScreen to obtain lat and lon coordinates in HGS coords
            with SphericalScreen(observer):
                hgs_coord = ls.centre_coord.transform_to(HeliographicStonyhurst)
            latitude = hgs_coord.lat.to(u.deg).value
            longitude = hgs_coord.lon.to(u.deg).value
                
        properties.append({
            't_obs'            : ls.submap.fits_header['T_OBS'],
            'class'            : ls.label_class,
            'confidence'       : ls.confidence,
            'Tx_arcsec'        : Tx,
            'Ty_arcsec'        : Ty,
            'box_w_pix'        : ls.box_w_pix,
            'box_h_pix'        : ls.box_h_pix,
            'x_centre_pix'     : ls.x_centre_pix,
            'y_centre_pix'     : ls.y_centre_pix,
            'longitude'        : longitude, 
            'latitude'         : latitude,
            'mean_intensity'   : mean_intensity,
            'max_intensity'    : max_intensity,
            'total_flux'       : total_flux,
            'area'             : area,
            'height'           : height,
            'width'            : width,
            'com_Tx'           : com_Tx,
            'com_Ty'           : com_Ty,
            'radial/tangential': ratio,
            'RHT coverage'     : rht_coverage,
            })    
    return properties

# =============================================================================
# Output Function
# =============================================================================
def write_output(properties, output_filename):
    
    with open(output_filename,"w", newline='') as output:
        writer = csv.DictWriter(output, fieldnames=properties[0].keys())
        writer.writeheader()
        writer.writerows(properties)


# =============================================================================
# Main Postprocessing Function
# =============================================================================
def postprocess(fits_dir: Path, label_dir: Path) -> None:
    
    fits_files = sorted(fits_dir.glob("*.fits"))
    label_files = {p.stem: p for p in Path(label_dir).glob('*.txt')}
    
                
    # make output params directory next to labels        
    output_dir = label_dir.parent / "params"
    if not output_dir.exists():       
        output_dir.mkdir(parents=True)      

    # Get first and last map to calculate time range
    first_map  = sunpy.map.Map(fits_files[0])
    last_map   = sunpy.map.Map(fits_files[-1])
    start_time = first_map.date - 12 * u.h
    end_time   = last_map.date + 12 * u.h

    #--- Get Pointing table ---
    pointing_table = get_pointing_table(source="JSOC", time_range=(start_time, end_time))

    # --- Get Correction table ---
    correction_table = get_correction_table("JSOC")


    for fits_file in tqdm(fits_files, desc="Processing FITS files", unit="file"):
        label_file = label_files.get(fits_file.stem)  # lookup matching label file
    
        if label_file is None:
            tqdm.write(f"No label file for {fits_file.name}, skipping")
            continue
        
        # Extract filename, converting to '.txt' as Path object
        filename = fits_file.with_suffix('.csv').name    
        output_filepath = output_dir / filename
                
        # Load in label txt file as numpy array & ensure 2D for single-line files
        labels = np.atleast_2d(np.loadtxt(label_file))
        
        # load in smap and remove disk
        smap = sunpy.map.Map(fits_file)
        
        # Calibrate
        smap_calibrated = aia_calibrated_map(smap, pointing_table, correction_table)
        
        # # Return LabelledSubclass dataclass for each label
        labelled_submaps = label_submaps(labels, smap_calibrated)
        properties       = calculate_box_properties(labelled_submaps)
        #print(properties)
        
        # Write output
        write_output(properties, output_filename=output_filepath)