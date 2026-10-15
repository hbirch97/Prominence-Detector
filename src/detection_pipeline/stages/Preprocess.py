#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:39:21 2026

@author: Harry Birch

Preprocessing Code

Takes FITs file input from given directory.

Processes in format for Prominence_AR model

Returns 3 channel tensor with channels:
    1. Grayscale
    2. WOW-processed grayscale
    3. Grayscale with the solar disk removed

"""


import astropy.units as u
from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta
from pathlib import Path
import sunpy.map
import os
import re
import json
from tqdm import tqdm 
import numpy as np
from PIL import Image
from skimage.transform import resize
from astropy.coordinates import SkyCoord 
from sunpy.map.maputils import all_coordinates_from_map ,coordinate_is_on_solar_disk 

# Function to remove the solar disk from a given smap
def remove_disk(smap, mask, data=None, size=None):       
        """
    

    Parameters
    ----------
    smap : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    data : TYPE, optional
        DESCRIPTION. The default is None.
    size : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
        # If a specific size is provided, resize the mask accordingly
        if size is not None:
            mask = resize(mask, size)
        
        # if data is provided applies maks 
        if data is not None:
            return data * ~mask
        
        # return smap with disk removed
        smap = smap.data * ~mask  
        return smap 

# Calibration function for AIA data
def aia_calibrated_map(smap, pointing_table, correction_table):
    """
    

    Parameters
    ----------
    smap : TYPE
        DESCRIPTION.
    pointing_table : TYPE
        DESCRIPTION.
    correction_table : TYPE
        DESCRIPTION.

    Returns
    -------
    smap : TYPE
        DESCRIPTION.

    """
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

# Create filename based on FITs Header.
# Added case to deal with SUVI data
def create_filename(smap):
    """
    

    Parameters
    ----------
    smap : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    name : TYPE
        DESCRIPTION.

    """
    meta = smap.meta
    
    # Check for wavelength
    wl = None
    for key in ["wavelnth", "wavelen", "wavel"]:
        if key.upper() in meta:
            wl = int(meta[key.upper()])
            break
    if wl is None:
        raise ValueError("No wavelength found in FITS header.")
    
    # Extract observation time
    if "DATE-OBS" not in meta:
        raise ValueError("DATE-OBS missing from header.")
    
    date_str = meta["DATE-OBS"]
    # If fractional seconds exist and are 1–3 digits, pad to microseconds (6 digits)
    if "." in date_str:
        base, frac = date_str.split(".")
        # keep only digits; pad right to 6 microsecond digits
        frac = (frac.rstrip("Z") + "000000")[:6]
        date_str = f"{base}.{frac}"

    dt = datetime.fromisoformat(date_str.replace("Z", ""))  # ISO safe parse
    
    # AIA fractional seconds (two digits)
    frac = f"{dt.microsecond//10000:02d}"
    
    # Find Instrument
    inst = meta.get("INSTRUME", "").lower()
    
    if "aia" in inst:
        prefix = "aia.lev1"
    elif "suvi" in inst:
        prefix = "suvi.lev2"
    else:
        prefix = inst.replace(" ", "_") or "unknown"
    
    # Create filename
    name = (
        f"{prefix}."
        f"{wl}A_"
        f"{dt.year:04d}_{dt.month:02d}_{dt.day:02d}"
        f"T{dt.hour:02d}_{dt.minute:02d}_{dt.second:02d}"
        f".{frac}Z.image_{prefix.split('.')[-1]}"
    )
    
    return name
    
# Function to apply log transformation to grayscale image (array)
def log_grayscale(arr):
    """
    

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    arr_log : TYPE
        DESCRIPTION.

    """
    
    # ensure array is clipped to minimum value of 0 to avoid negatives
    arr = arr.clip(0)
    
    # Calculate the 1st percentile value of the array to handle background levels
    vmin = np.percentile(arr, [1])

    # Apply log transformation
    arr_log = np.log10(arr - vmin + 1e-6)
    
    # Clip values to prevent any negative log results
    arr_log = np.clip(arr_log, 0, None)

    # Return log transformed array
    return arr_log


# Normalising function, depending on type of image transformation.
# Rescales resulting array to range [0, 255] (uint8 format)
def normalise(arr, percentiles, mode=None):
    """
    

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    percentiles : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

    """
    
    match mode:
        case None:        
            # Linear normalization using provided percentiles (default mode)
            arr = np.clip(arr, percentiles[0], percentiles[1])
            arr = (255 * (arr - percentiles[0]) / (percentiles[1] - percentiles[0])).astype(np.uint8)
            
        case "log":
            # Log normalisation mode
            arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
            arr_uint8 = (255 * arr_norm).astype(np.uint8)
            arr = arr_uint8
            
        case "log-stretch" |"stretch":                        
            # Log-stretch mode (depreciated)
            p1, p99 = np.percentile(arr, [1, 99])
            L_scaled = np.clip((arr - p1) / (p99 - p1), 0, 1)
            arr = (255 * L_scaled).astype(np.uint8)
    
    # Returns normalised array
    return arr


    
def preprocess(input_dir: Path,  output_dir: Path, wow_dir: Path, run_name: str) -> None:
    
    
    """
    Parameters
    ----------
    input_files : LIST
        List of FITs files to be converted into the test image type. If in directory, load in using:
            
            from pathlib import Path
            dir_path = Path("/path/to/your/directory")
            file_paths = [str(file) for file in dir_path.iterdir() if file.is_file()]

            
    save_path : str
        Save path for output images
        
    Returns
    -------
    Saves each converted FITs file as a 640x640 jpg images stored into given directory

    """
    from aiapy.calibrate.util import get_pointing_table, get_correction_table
        
    # Load WOW module
    os.chdir(wow_dir)
    from watroo import wow
    
    os.chdir('/home/w22043051/Documents/Project Work/Code/ML/ProminenceAR_Model/GitHub_Files/')
    
    # Make output dir if does not exists
    dst = output_dir / run_name / "preprocessed_images"
    if not dst.exists():       
        dst.mkdir(parents=True)  

    # Sort the input_files
    input_files = sorted(input_dir.glob("*.fits"))
    print(f"\nProcessing {len(input_files)} files")

    # Get first and last map to calculate time range
    first_map = sunpy.map.Map(input_files[0])
    last_map = sunpy.map.Map(input_files[-1])

    start_time = first_map.date - 12 * u.h
    end_time = last_map.date + 12 * u.h

    print(f"start time: {start_time}")
    print(f"end time: {end_time}")
    
    #--- Get Pointing table ---
    print("\nFetching pointing table")
    pointing_table = get_pointing_table(source="JSOC", time_range=(start_time, end_time))
    print("done!\n")

    # --- Get Correction table ---
    print("Fetching correction table")
    correction_table = get_correction_table("JSOC")
    print("done!\n")
    

    # Create percentile variables for each channel for later scaling
    percentiles = None
    wow_percentiles = None
    nodisk_percentiles = None
    
    # --- 3-Channel Data Processing ---
    
    # Process each file in input_files to 3-channel format
    for filename in tqdm(input_files, desc="Converting FITS files", unit="file"):
        try:
            
            #  --- Load raw SunPy map ---
            print('Making smap')
            smap = sunpy.map.Map(filename)
            
            # --- Create output filename ---
            outname = Path(filename).with_suffix(".jpg").name

            # --- Calibrate (geometric + radiometric) ---
            if smap.data.shape == (4096,4096):
                print('Calibrating')
                smap_calibrated = aia_calibrated_map(smap, pointing_table, correction_table)
            else:
                smap_calibrated = smap
            # Downscale output
            smap_calibrated_resized = resize(smap_calibrated.data, (640, 640))
            
            # Calculate mask for disk removal
            hpc_coords = all_coordinates_from_map(smap_calibrated) 
            mask = coordinate_is_on_solar_disk(hpc_coords) 

            # --- Grayscale intensity scaling ---
            arr_grayscale = smap_calibrated_resized.data
            if percentiles is None:
                percentiles = (np.float64(-0.0527532641561405), np.float64(74.66443487341849))
                arr_grayscale = normalise(arr_grayscale, percentiles)
            else:
                arr_grayscale = normalise(arr_grayscale, percentiles)
            
            # --- Remove Disk ---
            print('Removing disk')
            smap_nodisk = remove_disk(smap_calibrated, mask)
            smap_nodisk = resize(smap_nodisk, (640, 640))
            arr_nodisk = smap_nodisk.data
            if nodisk_percentiles is None:

                nodisk_percentiles = (np.float64(-0.0527532641561405), np.float64(26.183702169888555))
                arr_nodisk = normalise(arr_nodisk, percentiles=nodisk_percentiles)
            else:
                arr_nodisk = normalise(arr_nodisk, percentiles=nodisk_percentiles)
            
            # --- Apply WOW transform ---
            print('Applying WOW')
            smap_wow, _ = wow(smap_calibrated_resized, bilateral=1, denoise_coefficients=[5, 2])
            # Check for NANs
            if np.any(np.isnan(smap_wow)):
                arr_wow = np.nan_to_num(smap_wow, nan=0.0)
                print(f"{filename} has NANs!")
            else:
                arr_wow = smap_wow
                
            if wow_percentiles is None:
                wow_percentiles = (np.float64(-6.149568562819768), np.float64(13.720967562959041))
                arr_wow = normalise(arr_wow, percentiles=wow_percentiles)
            else:
                arr_wow = normalise(arr_wow, percentiles=wow_percentiles)
            
            # --- Convert to 3-channel RGB image ---
            print("Creating 3-channel image")
            arrays = [arr_grayscale, arr_wow, arr_nodisk]
            arr = np.stack((arrays[0], arrays[1], arrays[2]), axis=-1)
            img = Image.fromarray(arr, mode="RGB")
    
            # --- Flip vertically to fix origin ---
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
            # --- Save as JPG ---
            outpath = os.path.join(dst, outname)      
            try:
                img.save(outpath, quality=95)
                if os.path.isfile(outpath):
                    print(f"File successfully saved at: {outpath}")
                else:
                    print(f"Save attempted, but file not found at: {outpath}")
            except Exception as e:
                print(f"Error saving file: {e}")
                
        except Exception as e:
            tqdm.write(f"Failed to process {filename}: {e}")


