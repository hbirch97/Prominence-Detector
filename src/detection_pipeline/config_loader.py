#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:51:42 2026

@author: Harry Birch
config loader function
"""

import yaml
from pathlib import Path
from config import PipelineConfig, DRMSConfig


def load_config(path: str | Path) -> PipelineConfig:
    """
    Loads pipeline configuration from config YAML file.
    
    Reads the configuration file at the specified path and converts its
    contents into a PipelineConfig object. Nested download configuration
    is converted into a DRMSConfig object, and directory paths are
    converted from strings into pathlib.Path objects.
    
    Args:
        path (str | Path): Path to the YAML configuration file.
    
    Returns:
        PipelineConfig: A pipeline configuration object populated with
            the values from the YAML file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Convert nested download config into a DRMSConfig object
    if data.get("download") is not None:
        data["download"] = DRMSConfig(**data["download"])
    
    return PipelineConfig(**data)