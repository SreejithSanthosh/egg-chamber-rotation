"""Display a tracked dataset and take user input for updating it."""

# Import packages
import os
import sys
import numpy as np
from imageio.v3 import imread
import napari
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

# Add the tools directory to Python's path list and import imtools
sys.path.append('../code/')
from functions.track import TrackedTimelapse
from functions.utils import (
    flag_discontinuous_labels,
    make_centroid_points_layer,
    resegment_tt_in_viewer,
    select_layer,
)

# Hard code the path to the example time-lapse
DATA_DIR = ("/Users/sierraschwabach/Documents/All_tracks/new_rot")
BASENAME = "Exp34apical_06"









# UI parameters
T_STEPS_TO_RESEGMENT = 10

# Import the images and convert to an array
ims_intensities_path = os.path.join(DATA_DIR, f"{BASENAME}.tif")
ims_intensities = imread(ims_intensities_path)
#ims_intensities_reg_path = os.path.join(DATA_DIR, f"{BASENAME}_reg.tif")
try:
    ims_intensities_reg = imread(ims_intensities_reg_path)
    tt = TrackedTimelapse(ims_intensities_reg, ims_mask=None, basename=BASENAME, out_dir=DATA_DIR)

except:
    print("No registered intensities timelapse found")
    tt = TrackedTimelapse(ims_intensities, ims_mask=None, basename=BASENAME, out_dir=DATA_DIR)

# Initiate viewer, then make an intensities layer and tracked layer.
# Uses registered timelapse if available. 
viewer = napari.Viewer()
try:
    intensities_layer = viewer.add_image(ims_intensities_reg, name="intensities registered")
except:
    intensities_layer = viewer.add_image(tt.ims_intensities, name="intensities")
tracked_layer = viewer.add_labels(
    tt.ims_tracked, name="tracked", blending="additive", opacity=0.40
)

# Make a points layer to display region labels
centroid_points_layer = make_centroid_points_layer(viewer, tracked_layer)
select_layer(viewer, tracked_layer)


@viewer.bind_key("r")
def resegment_few(viewer):
    """Resegment the current t and some subsequent frames too."""
    resegment_tt_in_viewer(viewer, tracked_layer, tt, T_STEPS_TO_RESEGMENT)


@viewer.bind_key("t")
def resegment_the_rest(viewer):
    """Resegment the current t and all subsequent frames too."""
    resegment_tt_in_viewer(viewer, tracked_layer, tt, tt.t_total)


@viewer.bind_key("0")
def set_label_to_zero_select_fill(viewer):
    """Set up the GUI to delete regions."""
    tracked_layer.selected_label = 0
    tracked_layer.mode = "FILL"


@viewer.bind_key("9")
def new_label_select_paintbrush(viewer):
    """Set up the GUI to paint with a fresh label."""
    tracked_layer.selected_label = tracked_layer.data.max() + 1
    tracked_layer.mode = "PAINT"


# Initialize an empty table widget
label_table = QTableWidget()
label_table.setColumnCount(3)
label_table.setHorizontalHeaderLabels(("time", "label", "n"))
viewer.window.add_dock_widget(label_table)


@viewer.bind_key("8")
def check_for_discontinuous_regions(viewer):
    """Check through all timepoints for discontinous region labels."""
    msg = f"Checking all timepoints for discontinuous regions..."
    viewer.status = msg
    flag_labs = flag_discontinuous_labels(tracked_layer.data)
    if flag_labs is None:
        label_table.setRowCount(0)
    else:
        label_table.setRowCount(flag_labs.shape[0])
        for row in range(flag_labs.shape[0]):
            for col in range(3):
                label_table.setItem(
                    row, col, QTableWidgetItem(str(flag_labs[row, col]))
                )


# Change viewer title
viewer.title = f"Correcting segmentation and tracking for {BASENAME}"
"""
def layer_change_callback(event):
    print('a layer changed')
viewer.layers.events.inserted.connect(layer_change_callback)
"""

# Launch GUI
napari.run()
