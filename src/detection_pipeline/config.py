#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:40:47 2026

@author: Harry Birch
"""


from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class DRMSConfig:
    email: str
    start: str
    end: str
    wavelength: int
    cadence: str


@dataclass
class PipelineConfig:
    # required
    output_dir: Path
    run_name: str
    
    wow_dir: Path
    weights: str
    conf: float
    
    # database
    spatial_iou_threshold: float = 0.25
    temporal_iou_threshold: float = 0.5
    event_persistence_cadence: str = "0h"
    
    # optional
    download: DRMSConfig | None = None

