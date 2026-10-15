#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:17:25 2026

@author: Harry Birch

Function to download SDO/AIA data using drms
"""

import os
import drms
import time
from pathlib import Path
os.chdir('/home/w22043051/Documents/Project Work/Code/ML/ProminenceAR_Model/GitHub_Files/')
from config import DRMSConfig, PipelineConfig


    
def drms_download(config: DRMSConfig, output_dir: Path, run_name: str) -> None:
    """
    drms downloader    
    
    Parameters
    ----------
    config : DRMSConfig
        dataclass object for the download parameters.
    output_dir : str
        output directory specified in the config file.
    run_name : str
        output run name specified in the config file.

    Returns
    -------
    None
    
    Exports downloaded fits files in output_dir/run_name/

    """
    # drms config parameters
    start = config.start
    end = config.end
    wavelength = config.wavelength
    cadence = config.cadence
    email = config.email # email must be registered for JSOC exports at: http://jsoc.stanford.edu/ajax/register_email.html

    # Make output dir if does not exists
    dst = output_dir / run_name / "downloads"
    if not dst.exists():       
        dst.mkdir(parents=True)   
        

    # downloading the data
    start_time = time.time()
    client = drms.Client()

    qstr = 'aia.lev1_euv_12s['+start+'-'+end+'@'+cadence+']['+str(wavelength)+'][? QUALITY = 0 ?]{image}'         
    print(f"Data export query:\n  {qstr}\n")

    # Submit export request using the 'url-tar' method, protocol default: 'as-is'
    print("Submitting export request...")
    result = client.export(qstr, method="url", protocol='fits', email=email)

    # Print request URL.
    print(f"\nRequest URL: {result.request_url}")
    print(f"{len(result.urls)} file(s) available for download.\n")
    
    # Download selected files.
    dr = result.download(dst)
    print("Download finished.")
    print(f'\nDownloaded file:\n  "{dr.download[0]}"\n')
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
