#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:50:21 2026

@author: Harry Birch

Main code body to run all stages of the pipeline
[1] Optional downloading
[2] Preprocessing imagesd
[3] Inference with the YOLOv11 prominence-AR model
[4] Postprocessing of the label data
[5] Compiling the resulting event database
"""
from pathlib import Path

from config import DRMSConfig, PipelineConfig
from config_loader import load_config

from stages.Download import drms_download
from stages.Inference import inference
from stages.Preprocess import preprocess
from stages.Postprocess import postprocess
from stages.Database import create_database



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

config = load_config(CONFIG_PATH)

print(config.run_name)
print(config.output_dir)
print(config.download.email)

# %%


def run_pipeline(config: PipelineConfig):
         
    dst = config.output_dir / config.run_name
    if not dst.exists():       
        Path(dst).mkdir(parents=True)   
    
    
    if config.download:
        print("\n[1/5] === Downloading data ===")
        drms_download(
            config=config.download,
            output_dir=config.output_dir,
            run_name=config.run_name,
        )
        
    print("\n[2/5] === Preprocessing images ===")
    preprocess(input_dir=(config.output_dir / config.run_name / "downloads"),
               output_dir=config.output_dir,
               wow_dir=config.wow_dir,
               run_name=config.run_name)

    print("\n[3/5] === Running inference ===")
    inference(input_dir=(config.output_dir / config.run_name / "preprocessed_images"),
              output_dir=config.output_dir,
              run_name=config.run_name,
              conf=config.conf)
                
    print("\n[4/5] === Postprocessing label data ===")
    postprocess(fits_dir=(config.output_dir / config.run_name / "downloads"),
                label_dir=(config.output_dir / config.run_name / f"inference_conf={config.conf}/labels")
                )
    
    print("\n[5/5] === Compiling Database ===")
    create_database(
        csv_dir=(config.output_dir / config.run_name / f"inference_conf={config.conf}/params"),
        cadence=config.download.cadence if config.download is not None else None,
        spatial_iou_threshold=config.spatial_iou_threshold,
        temporal_iou_threshold=config.temporal_iou_threshold,
        event_persistence_cadence=config.event_persistence_cadence
        )
    
run_pipeline(config)