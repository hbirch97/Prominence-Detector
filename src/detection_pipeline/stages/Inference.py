#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:48:18 2026

@author: Harry Birch

Inference code for detecting prominences and ARs from 3-channel composite images
"""

#pip install huggingface_hub
from pathlib import Path
import os

def inference(input_dir: Path, output_dir: Path, run_name: str, conf: float) -> None:

    
    from huggingface_hub import hf_hub_download # import model weights from huggingface
    from ultralytics import YOLO # import model from ultralytics 
    
    
    weights_path = hf_hub_download(
        repo_id="hbirch97/Prominence-AR_Detector",
        filename="Prominence-AR_weights.pt",
        cache_dir=Path("./cache").resolve()
    )
                
    # Load trained model
    model = YOLO(weights_path)

    # Run inference on test set
    results = model.predict(
        source=input_dir,
        project=(output_dir / run_name),
        name=f"inference_conf={conf}",
        conf=conf,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    