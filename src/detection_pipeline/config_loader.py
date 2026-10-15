#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:51:42 2026

@author: w22043051
"""
import pandas as pd
import yaml
from pathlib import Path
from config import PipelineConfig, DRMSConfig


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Convert nested download config into a DRMSConfig object
    if data.get("download") is not None:
        data["download"] = DRMSConfig(**data["download"])

    # Convert directory strings into Path objects
    data["output_dir"] = Path(data["output_dir"])
    data["wow_dir"] = Path(data["wow_dir"])
    
    return PipelineConfig(**data)


#config = load_config("/home/w22043051/Documents/Project Work/Code/ML/ProminenceAR_Model/GitHub_Files/configs/config.yaml")

# %%
# def __post_init__(self):
#     for name in (
#         "spatial_iou_threshold",
#         "temporal_iou_threshold",
#     ):
#         value = getattr(self, name)
#         if not 0 <= value <= 1:
#             raise ValueError(
#                 f"{name} must be between 0 and 1, got {value}"
#             )

#     # Validate the duration string
#     try:
#         duration = pd.Timedelta(self.event_persistence_cadence)
#     except (ValueError, TypeError):
#         raise ValueError(
#             "event_persistence_cadence must be a valid duration, "
#             "e.g. '6h', '30m', or '0h'"
#         ) from None

#     if duration < pd.Timedelta(0):
#         raise ValueError(
#             "event_persistence_cadence must be non-negative"
#         )