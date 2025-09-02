"""
Measure tissue migration rate from tracked cells.

For small egg chambers, uses all tracked cells. For large egg chambers,
uses just cells in a medial region.

Finds net displacement vector of all included cells in all pairs of consecutive
frames. Migration speed is the length of this vector.

Also outputs the medial mask and the tracked cells used for measurement.
"""

import os
import numpy as np
from imageio import volread, volwrite
import pandas as pd
from skimage.measure import label, regionprops

# Internal functions
from functions.utils import select_files, cart_to_pol
from functions.segment import select_mask_adjacent, tissue_axis_mask
from functions.measure import tissue_medial_orientation

# Set data location, parameters
DATA_DIR = ("/Volumes/sierra/migration_speeds_setup/10930_abi_ECs_tifs/new/diff_scaling/")
OUT_DIR = DATA_DIR
FRAMES_PER_MIN = 2
PIXELS_PER_UM = 0.6411
MEDIAL_REGION_WIDTH_UM = 25 # width of medial mask along A-P (long) axis, in um.
MEDIAL_TO_TOTAL_RATIO = 2/3 # Threshold above which the whole tissue is used for rate measurement
MIN_TO_ANALAYZE = 10

medial_region_width_px = int(MEDIAL_REGION_WIDTH_UM * PIXELS_PER_UM)
frames_to_analyze = MIN_TO_ANALAYZE * FRAMES_PER_MIN + 1

# Get paths to timelapses
datasets = select_files(DATA_DIR, [".tif", "_tissue_mask.tif", "_tracked.tif"])

for dataset in datasets:
    # Import timelapse data
    ims_tracked = volread(dataset["_tracked.tif"])
    tissue_masks = volread(dataset["_tissue_mask.tif"])
    # Save in dataset dictionary
    dataset["ims_tracked"] = ims_tracked
    dataset["tissue_masks"] = tissue_masks
    # Measure tissue long axis length
    tissue_mask_lab = label(tissue_masks[0])
    props = regionprops(tissue_mask_lab)[0]
    major_axis_length = props["major_axis_length"]

    # Make medial region masks (or copy the tissue masks)
    if major_axis_length <  medial_region_width_px / MEDIAL_TO_TOTAL_RATIO:
        # Tissue is too short. Do not exclude any cells.
        dataset["masked_medial"] = "No"
        medial_masks = np.copy(tissue_masks)
    else:
        # Tissue is long enough. Make a mask that includes just medial cells.
        dataset["masked_medial"] = "Yes"
        tissue_masks_bool = tissue_masks.astype(bool)
        # Find the tissue centroid, medial orientation in the first frame
        medial_ori = tissue_medial_orientation(tissue_masks_bool[0])
        centroid = props.centroid
        # Make medial masks for each frame. Uses the first frame orientation
        # and centroid throughout the timelapse.
        medial_masks = np.zeros_like(tissue_masks, dtype=bool)
        for t in range(len(tissue_masks)):
            medial_masks[t] = tissue_axis_mask(
                tissue_masks_bool[t], medial_region_width_px, medial_ori, centroid)

    # Make timelapse with only tracked cells that are entirely within medial_masks
    ims_medial = np.zeros_like(ims_tracked)
    for t in range(len(ims_tracked)):
        medial_interior_mask = medial_masks[t] * ~select_mask_adjacent(ims_tracked[t], medial_masks[t])
        ims_medial[t] = ims_tracked[t] * medial_interior_mask

    # Store results in dictionary
    dataset["medial_masks"] = medial_masks
    dataset["ims_medial"] = ims_medial

# Calculate migration speed for each dataset
basenames = []
speeds = []
masked_medial = []
for dataset in datasets:
    ims_medial = dataset["ims_medial"]

    # Find cell centroids
    df_centroids_ls = []
    for t, im_medial in enumerate(ims_medial[:frames_to_analyze]):
        labs = []
        centroid_rows = []
        centroid_cols = []
        for region in regionprops(im_medial):
            lab = region.label
            labs.append(lab)
            centroid_row, centroid_col = region.centroid
            centroid_rows.append(centroid_row)
            centroid_cols.append(centroid_col)

        df_frame = pd.DataFrame({"cell":labs,
                                 "frame":t,
                           "centroid_row":centroid_rows,
                           "centroid_col":centroid_cols})
        df_centroids_ls.append(df_frame)

    # Find displacements for each cell between consecutive frames
    df_disps_list = []
    for t in range(1,len(df_centroids_ls)):
        # Find cells present in frames t-1 and t
        cells_t0 = list(df_centroids_ls[t-1]['cell'])
        cells_t1 = list(df_centroids_ls[t]['cell'])
        cells_t0_set = set(cells_t0)
        shared_cells_set = cells_t0_set.intersection(cells_t1)
        shared_cells_list = list(shared_cells_set)

        r_disps = []
        c_disps = []
        dir_disps = []
        mag_disps = []
        for i in range(len(shared_cells_list)):
            cell = shared_cells_list[i]
            df_t0 = df_centroids_ls[t-1]
            df_t1 = df_centroids_ls[t]
            r_t0 = float(df_t0[df_t0['cell']==cell]['centroid_row'])
            c_t0 = float(df_t0[df_t0['cell']==cell]['centroid_col'])
            r_t1 = float(df_t1[df_t1['cell']==cell]['centroid_row'])
            c_t1 = float(df_t1[df_t1['cell']==cell]['centroid_col'])
            r_disp = (r_t1 - r_t0) / PIXELS_PER_UM * FRAMES_PER_MIN
            c_disp = (c_t1 - c_t0) / PIXELS_PER_UM * FRAMES_PER_MIN
            dir_disp, mag_disp = cart_to_pol(c_disp, r_disp)
            r_disps.append(r_disp)
            c_disps.append(c_disp)
            dir_disps.append(dir_disp)
            mag_disps.append(mag_disp)

        df_disps_frame = pd.DataFrame({"cell":shared_cells_list,
                                        "r_disp":r_disps,
                                        "c_disp":c_disps,
                                        "dir_disp":dir_disps,
                                        "mag_disp":mag_disps,
                                        "frame_start":t-1,
                                        "frame_end":t})
        df_disps_list.append(df_disps_frame)

    df_disps = pd.concat(df_disps_list)
    r_disp_mean = np.mean(df_disps['r_disp'])
    c_disp_mean = np.mean(df_disps['c_disp'])
    dir_mean, speed_mean = cart_to_pol(c_disp_mean, r_disp_mean)

    basenames.append(dataset["basename"])
    speeds.append(speed_mean)
    masked_medial.append(dataset["masked_medial"])

# Make dataframe of migration rate data, output as CSV
df_migration = pd.DataFrame({"sample":basenames,
                             "masked_medial_cells":masked_medial,
                             "mig_speed_um_per_min":speeds})

out_path = os.path.join(OUT_DIR, "migration_rate.csv")
df_migration = df_migration.sort_values(['sample'])
df_migration.reset_index(inplace=True, drop=True)
df_migration.to_csv(path_or_buf = out_path)

# Output medial_masks, ims_medial as tif stacks
for dataset in datasets:
    basename = dataset["basename"]
    medial_masks = dataset["medial_masks"].astype('uint8')
    ims_medial = dataset["ims_medial"]
    medial_masks_path = os.path.join(OUT_DIR, f'{basename}_medial_mask.tif')
    ims_medial_path = os.path.join(OUT_DIR, f'{basename}_tracked_medial.tif')
    volwrite(medial_masks_path, medial_masks)
    volwrite(ims_medial_path, ims_medial)
