#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:40:47 2026

@author: Harry Birch
"""


from dataclasses import dataclass
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
print(f"config code root = {PROJECT_ROOT} \n _____________ \n")

# Function to return absolute path from given relative path.
# uses PROJECT_ROOT 
def _abs(p, base=PROJECT_ROOT):
    p = Path(p).expanduser() # convert to Path if not already with home directory
    return p if p.is_absolute() else (base / p).resolve()


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
    conf: float
    
    # database
    spatial_iou_threshold: float = 0.25
    temporal_iou_threshold: float = 0.5
    event_persistence_cadence: str = "0h"
    
    # optional
    download: DRMSConfig | None = None

    def __post_init__(self):
        # resolve paths
        self.output_dir = _abs(self.output_dir)
        if self.wow_dir is not None:
            self.wow_dir = _abs(self.wow_dir)
            
        # Validate thresholds
        for name in (
            "spatial_iou_threshold",
            "temporal_iou_threshold",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1, got {value}"
                )

        # Validate duration
        try:
            duration = pd.Timedelta(
                self.event_persistence_cadence
            )
        except (ValueError, TypeError):
            raise ValueError(
                "event_persistence_cadence must be a valid "
                "duration, e.g. '6h', '30m', or '0h'"
            ) from None

        if duration < pd.Timedelta(0):
            raise ValueError(
                "event_persistence_cadence must be non-negative"
            )