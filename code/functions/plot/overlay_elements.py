"""
Functions overlaying visible elements on an image.

These can be stacked up to produce more complex figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation
from skimage.measure import regionprops, label
from ..segment import select_in_field
from ..utils import validate_mask


def overlay_centroids(
    im, ax, mask=None, periphery_excluded=True, color=(1, 1, 1), alpha=1, size=2
):
    """
    Plot dots on region centroids on an Axes object.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    ax : matplotlib Axes object
        Where centroids will be plotted
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether regions touching the border or mask
        should be included in the plot
    color : 3-element tuple
        RGB color for dots
    alpha : float
        Transparency from 0 to 1
    size : int
        Size of each dot, as passed to matplotlib's scatter function

    Returns
    -------
    artist : matplotlib Artist
        output by plt.scatter
    """
    mask = validate_mask(im, mask)

    # Apply mask by setting masked pixels to zero
    im_masked = np.copy(im)
    im_masked[mask == False] = 0

    # Set mask- and border-adjacent regions to zero
    if periphery_excluded:
        im_masked = im_masked * select_in_field(im, mask)

    # Get properties of the labeled regions
    centroid_list = []
    for region in regionprops(im_masked):
        centroid_row, centroid_col = region.centroid
        centroid_list.append(np.array((centroid_col, centroid_row)))

    # Stack as one array
    centroid_array = np.stack(centroid_list)

    # Make a 2D array of RGBA colors
    color_array = np.ones((len(centroid_list), 4))
    rgba = np.append(np.array(color), alpha)
    color_array[:] = rgba

    # Plot points
    artist = ax.scatter(
        centroid_array[:, 0], centroid_array[:, 1], c=color_array, s=size
    )
    return artist


def overlay_edges(
    im, mask=None, periphery_excluded=True, thickness=4, color=(1, 1, 1), alpha=1
):
    """
    Generate an RGBA image of region edges.

    The output image is transparent except for the edges
    between labeled regions, which are highlighted.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether regions touching the border or mask
        should be included in the plot
    thickness : int
        Line thickness, plots closest even number
    color :  3-element tuple of floats
        Interpreted as RGB colors
    alpha : float
        Transparency from 0 to 1

    Returns
    -------
    im_rgba : ndarray with dimensions (y,x,4)
        Region edges highlighted with lines
    """
    mask = validate_mask(im, mask)

    # Apply mask by setting masked pixels to zero
    im_masked = np.copy(im)
    im_masked[mask == False] = 0

    # Set mask- and border-adjacent regions to zero
    if periphery_excluded:
        im_masked = im_masked * select_in_field(im, mask)

    # Find boundaries as boolean array
    boundaries = find_boundaries(im_masked)

    # Determine number of dilations needed to approximate thickness
    dilations = int(thickness / 2) - 1

    # Dilate boundaries
    for _ in range(dilations):
        boundaries = binary_dilation(boundaries)

    # Make an RGBA image
    rows, cols = np.shape(im_masked)
    im_rgba = np.zeros((rows, cols, 4))

    # Set the RBG channel values
    im_rgba[:, :, 0] = color[0]
    im_rgba[:, :, 1] = color[1]
    im_rgba[:, :, 2] = color[2]
    im_rgba[:, :, 3] = boundaries * alpha

    return im_rgba


def overlay_random_colors(
    im, mask=None, periphery_excluded=True, alpha=0.4, same_seed=True
):
    """
    Generate an RGBA image of regions.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether regions touching the border or mask
        should be included in the plot
    alpha : float
        Transparency from 0 to 1
    same_seed : bool
        If True, im_rgba colors will be consistent across repeated function
        calls for one input im

    Returns
    -------
    im_rgba : ndarray with dimensions (y,x,4)
        Regions labeled with random
    """
    mask = validate_mask(im, mask)

    # If periphery_excluded=True, set mask- and border-adjacent regions to zero
    if periphery_excluded:
        mask = mask * select_in_field(im, mask)

    # Make an array of unique cell labels
    im_relabeled = label(im * mask, background=-1) - 1
    unique_labels = np.unique(im_relabeled)

    # Make a set of random colors
    color_list = []

    for label_new in unique_labels:
        if same_seed == True:
            # find label in original image
            label_orig = np.unique(im * (im_relabeled == label_new))[-1]
            # Use the label as random seed
            np.random.seed(label_orig)
        elif same_seed == False:
            # No fixed seed
            pass
        color_list.append([np.random.rand() for i in range(3)] + [alpha])
    color_array = np.array(color_list)

    # Make an RGBA image
    rows, cols = np.shape(im_relabeled)
    im_rgba = np.zeros((rows, cols, 4))

    # Plot the colors on the original image
    im_rgba = color_array[im_relabeled]

    # Set masked regions to transparent
    im_rgba[:, :, 3][mask == False] = 0

    return im_rgba


def overlay_shape(im, color=(1, 1, 1), alpha=0.5):
    """
    Turn a 2D bool image into RGBA overlay image.

    Parameters
    ----------
    im : 2D bool ndarray
        No constraints on shape
    color : tuple of integers or floats
        RGB color for overlay
    alpha : float
        Transparency from 0 to 1

    Returns
    -------
    im_rgba : ndarray with dimensions (y,x,4)
        Each region filled with a different color
    """
    # Make an RGBA image
    rows, cols = np.shape(im)
    im_rgba = np.zeros((rows, cols, 4))

    # Set the RBG channel values
    im_rgba[:, :, 0] = color[0]
    im_rgba[:, :, 1] = color[1]
    im_rgba[:, :, 2] = color[2]
    im_rgba[:, :, 3] = im * alpha

    return im_rgba


def overlay_trait(
    im,
    data,
    mask=None,
    periphery_excluded=True,
    alpha=0.5,
    range_min=None,
    range_max=None,
    colormap=plt.cm.Spectral,
):
    """
    Create an RGBA image colored by intensity of an arbitrary trait.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    data : ndarray with dimensions (n,2)
        One row for each region. Column 0 is region labels,
        column 1 is trait values. If array has more than 2
        dimensions, the rest are ignored.
    mask : bool 2D ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether regions touching the border or mask should be included
        in the plot
    alpha : float
        Transparency from 0 to 1
    range_min : int
        The input value that is set to the lowest value in the colormap
    range_max : int
        The input value that is set to the highest value in the colormap
    colormap : matplotlib LinearSegmentedColormap object

    Returns
    -------
    im_rgba : ndarray with dimensions (y,x,4)
        Each region filled with a different color
    """
    mask = validate_mask(im, mask)
    # TODO: add select_in_field
    # TODO: finish This

    # Check shape of the data to be plotted
    if np.shape(data)[1] <= 1:
        raise ValueError("Shape of data should be (N,2)")

    # Make a transparent RGBA image
    rows, cols = np.shape(im)
    im_rgba = np.zeros((rows, cols, 4))

    # Need to fill this in here
    # Turn data values into an array of colors
    # Apply the colors to the image

    return im_rgba
