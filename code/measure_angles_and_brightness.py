#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import sys
from urllib.parse import urlparse
import numpy as np
from imageio import volread, volwrite, imwrite
from cellpose import models, utils, io, plot
import pandas as pd

# Post-processing after cellpose
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation, binary_erosion, disk


# Visualize output
import matplotlib.pyplot as plt
#import matplotlib as mpl
#matplotlib inline

sys.path.append('../code/')
from functions.utils import select_files, points_to_angle
from functions.segment import (largest_object_mask_timelapse, 
                               select_in_field, edge_between_neighbors, 
                               neighbor_array_nr,select_mask_adjacent, 
                               interface_endpoints_coords, 
                               neighbor_array_nr, 
                               cell_interiors_mask, 
                               cell_vertices_mask
                              )


# Set locations of images (intensities and segmented), output
DATA_DIR = ('/Users/sierraschwabach/Desktop/test_flat/')
OUT_DIR = ('/Users/sierraschwabach/Desktop/test_flat/')

# Set amount of edge dilation
EDGE_DIL_FACTOR = 5
VERTEX_DIL_FACTOR = 10 #used 10 for fat2 PCP fixed

# Set total channels in intensities image file,
# index of channel to measure (im_int),
# name of measured channel (for output naming)
CHANNELS_TOTAL = 2
INTENSITIES_CHANNEL_IND = 1
INTENSITIES_CHANNEL_NAME = "Fat2"

# Select files
file_info = select_files(DATA_DIR, ['.tif','_seg.npy'])
df_sample_ls = []
for i in range(len(file_info)):
    # Import intensities and segmented images
    im_intensities_path = file_info[i]['.tif']
    im_lab_path = file_info[i]['_seg.npy']
    if CHANNELS_TOTAL > 1:
        im_intensities_raw = volread(im_intensities_path)
        im_intensities = im_intensities_raw[INTENSITIES_CHANNEL_IND]
    else:
        im_intensities = imread(im_intensities_path)
    im_stack = np.load(im_lab_path, allow_pickle=True).item()
    im_lab = im_stack['masks']
    im_seg = im_stack['img']
    
      # Get condition and sample number
    basename = file_info[i]['basename']
    sample_num = basename.split('_')[-1]
    condition = basename.split('_' + sample_num)[0]
    
     # Track progress
    print(f'Analyzing image {str(i)} out of {str(len(file_info))}, {basename}')
    
#Post processing after cellpose
    #make tissue mask
    tissue_mask = im_lab > 0
    #Fill holes in tissue mask
    tissue_mask = binary_fill_holes(tissue_mask)
    #Fill in holes between cells with a watershed using labels as seeds
    im_lab = watershed(im_seg, 
                                markers=im_lab, mask=tissue_mask)
#Make tissue and cell labels masks
    # Select non-periphery cells
    tissue_mask_inbounds = select_in_field(im_lab, tissue_mask)
    im_lab_inbounds = im_lab * tissue_mask_inbounds   
  
    #Make mask of vertices (these get excluded)
    vertex_mask = cell_vertices_mask(im_lab, vertex_dilation_factor =
                                     VERTEX_DIL_FACTOR,
                                     periphery_excluded=False)
    
  # Find pairs of neighboring cells, exclude background
    neighbor_pairs_raw = neighbor_array_nr(im_lab)
    neighbor_pairs = neighbor_pairs_raw[neighbor_pairs_raw[:,1] > 0]
    
# Make lists where the current sample's data will be stored
    cell_a_ls = []
    cell_b_ls = []
    angles = []
    edge_masks = []
    edge_masks_trimmed = []

    for j in range(len(neighbor_pairs)):
        cell_a_mask = im_lab == neighbor_pairs[j][0]
        cell_b_mask = im_lab == neighbor_pairs[j][1]

        # Find vertices, angle between them, edge shape between them.
        # Skip cell pairs with more or less than 2 endpoints,
        # which happens occasionally
        try:
            # Get vertices, angle between them
            vertex_coords = interface_endpoints_coords(cell_a_mask,cell_b_mask)
            phi = points_to_angle(vertex_coords[0],vertex_coords[1])

            # Reflect angles so all are between 0 and pi/2
            if phi > np.pi/2:
                phi_quadrant = np.pi/2 - (phi - np.pi/2)
            else:
                phi_quadrant = phi
            angles.append(phi_quadrant)
            cell_a_ls.append(int(neighbor_pairs[j][0]))
            cell_b_ls.append(int(neighbor_pairs[j][1]))

            # Make edge mask, dilate, add to stack of edge masks
            edge = edge_between_neighbors(cell_a_mask, cell_b_mask)
            edge_dil_shape = disk(EDGE_DIL_FACTOR)
            edge_dil = binary_dilation(edge, footprint=edge_dil_shape)
            edge_masks.append(edge_dil)
            edge_masks_trimmed.append(edge_dil * ~vertex_mask)
        except:
            pass

    # Measure mean intensities
    # Measure mean intensity at cell interiors (the 'background')
    interiors_mask = cell_interiors_mask(im_lab_inbounds,
                                         edge_dilation_factor=EDGE_DIL_FACTOR)
    mean_interiors = np.mean(im_intensities[interiors_mask])

    # Measure mean intensity of each edge with or without vertices,
    # with or without background subtraction
    mean_edges = []
    mean_edges_minus_bkgd = []
    mean_edges_no_vertices = []
    mean_edges_minus_bkgd_no_vertices = []

    for j in range(len(edge_masks)):
        # With vertices
        mean_edge = np.mean(im_intensities[edge_masks[j]])
        mean_edge_minus_bkgd = mean_edge - mean_interiors

        # Without vertices
        if np.sum(edge_masks[j]) == 0: # case where edge is fully masked out
            mean_edge_no_vertices = np.nan
            mean_edge_minus_bkgd_no_vertices = np.nan
        else:
            mean_edge_no_vertices = np.mean(im_intensities[edge_masks_trimmed[j]])
            mean_edge_minus_bkgd_no_vertices = mean_edge_no_vertices - mean_interiors

        mean_edges.append(mean_edge)
        mean_edges_minus_bkgd.append(mean_edge_minus_bkgd)
        mean_edges_no_vertices.append(mean_edge_no_vertices)
        mean_edges_minus_bkgd_no_vertices.append(mean_edge_minus_bkgd_no_vertices)

    # Construct a sample df, add to list
    condition_ls = [condition]*len(cell_a_ls)
    sample_num_ls = [sample_num]*len(cell_a_ls)

    col_names = ['condition', 'sample_num', 'cell_a', 'cell_b', 'edge_angle_rad',
                 'mean_intensity', 'mean_int_minus_bkgd', 'mean_int_no_verts',
                 'mean_int_no_verts_minus_bkgd']
    df_sample = pd.DataFrame(list(zip(condition_ls, sample_num_ls, cell_a_ls,
                                    cell_b_ls, angles, mean_edges,
                                    mean_edges_minus_bkgd,
                                    mean_edges_no_vertices,
                                    mean_edges_minus_bkgd_no_vertices)),
                                columns = col_names)
    df_sample_ls.append(df_sample)

    # Temp
    df_sample_path = (OUT_DIR + basename + INTENSITIES_CHANNEL_NAME +
            '_edge_intensity_by_angle_sample.csv') 
    df_sample.to_csv(path_or_buf = df_sample_path)


# Concatenate sample dfs, sort, output as CSV
df_all = pd.concat(df_sample_ls)
df_sorted = df_all.sort_values(['condition', 'sample_num'])
df_sorted.reset_index(inplace=True, drop=True)
df_path = (OUT_DIR + INTENSITIES_CHANNEL_NAME +
            '_edge_intensity_by_angle_sample.csv')
df_sorted.to_csv(path_or_buf = df_path)
    

