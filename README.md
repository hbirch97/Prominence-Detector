
![Screenshot](banner.png)


# Prominence-Detector
Object detection model for prominences (and additionally active regions) built using YOLOv11. The repository includes code to construct a prominences database from a given set of observations, along with statistical analysis of the physical properties of each prominence event.

The code is streamlined specifically for limb prominences (white boxes in the below images) taken in 304 Å channel observations from SDO/AIA, however, the model is capable of generalising to other instruments, provided the input images are preprocessed in the correct format. The code is not intended for on-disk filaments (yellow arrows in the image), as many other routines exist for the automatic detection of these objects. 

<img src="Full_Disk.png" width="500">

For a background in the motivation and development of an earlier version of the model, please refer to this [article](https://iopscience.iop.org/article/10.3847/1538-4357/ae664d). 
An additional article describing the current model, processing pipeline, and solar cycle results is currently in preperation, and will be linked here once completed.

The outline of the code is as follows:
1. Optional FITS data downloading via DRMS. This is specificall for SDO/AIA 304 Å, although other sources may be used.
2. Preprocessing data into our specific 3-channel format as depicted in the [article](https://iopscience.iop.org/article/10.3847/1538-4357/ae664d). This requires downloading the [WOW GitHub repo](https://github.com/frederic-auchere/wow), ensuring the path is added to the config run file.
3. Model Inferece on the preprocessed data. The model weights are hosted on [HuggingFace](https://huggingface.co/hbirch97/Prominence-AR_Detector/blob/main/README.md) and are automatically cached during this process.
4. Postprocessing the inference labels using the original FITS file to extract relevant parameters (positional, geometrical, intensity, and structural classification). Each label file will have a corresponding params.csv file with the parameter information.
5. Database compilation, outputting as a single csv file.
6. **(TBD)** Analysis of the results, following from that of the article (in prep). An .ipynb notebook document will be provided to faciliate this.

Example inference image, using a preprocessed SDO/AIA 304 Å observation. The number after the class label denotes the prediction confidence.

<img src="example.jpg" width="500">

How to run this code:

1. install prerequisite packages from requirements.txt
```bash
pip install -r requirements.txt
```

2. edit the config.yaml file in configs/

Config
    output_dir: Path - output file directory. Defaults to ./output
    run_name: str - run name of folder within above output directory
    
    wow_dir: Path - path to WOW wavelets code
    conf: float = 0.4 - confidence threshold for the inference predictions. Predictions with lower confidence will be discarded. defaults to 0.4
    
    # database
    spatial_iou_threshold: float = 0.25 - spatial IoU threshold for filtering predictions for the database
    temporal_iou_threshold: float = 0.5 - temporal IoU threshold for filtering predictions for the tracking code
    event_persistence_cadence: str = "0h" - 
    
    # optional - must provide one
    download: DRMSConfig | None = None - optional download configs. If none supplied the code will expect to use FITS file in the ./output/run_name/downloads/ directory
    fits_dir: Path | None = None - if no download is supplied must supply instead a path to directory of fits files

Downloading configs:
    email: str - email string for registered JSOC email address
    start: str - start time for observations
    end: str - end time for observations 
    wavelength: int - wavelenght for observations (i.e. 304)
    cadence: str - observation cadence (i.e. 24h, 6m, 12s, etc.)

3. Run code as 
```python
python3 src/detection_pipeline/pipeline.py
```

4. Outputs are as follows:  
File tree:
```
[output_name]/
├── downloads/
│   └── *.fits
├── preprocessed_images/
│   └── *.jpg
├── inference_conf={conf}/
│   ├── labels/
│   │   └── *.txt
│   └── params/
│       └── *.csv
├── database.csv
└── analysis.ipynb
```
