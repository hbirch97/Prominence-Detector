#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:48:18 2026

@author: Harry Birch

Inference code for detecting prominences and ARs from 3-channel composite images
"""

from pathlib import Path

# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("Model device:", next(model.model.parameters()).device)

def inference(input_dir: Path, output_dir: Path, run_name: str, conf: float) -> None:
    """
    Run yolov11 inference on preprocessed images using the trained
    Prominence-AR detection model.

    Downloads the model weights from Hugging Face (using the local cache
    where available), loads the model using Ultralytics YOLO, and performs
    inference on the images in the specified preprocessed input directory.

    Prediction results, including annotated images and bounding box
    detections, are saved to /inference_conf={conf}. Detection
    confidence scores are also saved alongside the bounding box labels.

    Args:
        input_dir (Path): Directory containing the preprocessed images to run inference on.
        output_dir (Path): Parent directory where inference results will be saved.
        run_name (str): Name of the inference run, used to organise output files.
        conf (float): Minimum confidence threshold for retaining detections.

    Returns:
        None
    """
    
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
    model.predict(
        source=input_dir,
        project=str(output_dir / run_name),
        name=f"inference_conf={conf}",
        conf=conf,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        device="cpu",
        exist_ok=True
    )