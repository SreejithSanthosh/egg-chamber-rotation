"""
Segment cells in each frame of a timelapse, then track them.

Input: A tif timelapse of an egg chamber with cell edges labeled.

Output: A tif timelapse with labeled cells.
"""

# Import packages
import os
import sys
import numpy as np
from imageio.v3 import imread
from functions.track import TrackedTimelapse
from functions.utils import select_files

# Hard code the path to the timelapses
DATA_DIR = ("/Users/sierraschwabach/Documents/All_tracks/new_rot/")

# Get paths to intensities timelapses
datasets = select_files(DATA_DIR, [".tif"])
for dataset in datasets:
    basename = dataset["basename"]
    print(f"Segmenting and tracking {basename}")

    # Import the raw images and convert to an array
    ims_intensities = imread(dataset[".tif"])

    # Make a TrackedTimelapse object
    tt = TrackedTimelapse(ims_intensities, cell_diam=70, basename=basename, out_dir=DATA_DIR)
    
