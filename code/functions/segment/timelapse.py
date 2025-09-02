"""Functions for specific to timelapse datasets."""

import numpy as np
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from .tissue import (epithelium_watershed,
                    largest_object_mask,
                    segment_epithelium_cellpose)
from ..utils import validate_mask


def segment_epithelium_timelapse(
    ims_intensities, ims_mask=None, ims_corr=None, ims_seeds=None
):
    """
    Segment a timelapse of a live-imaged epithelium.

    Parameters
    ----------
    ims_intensities : 2D+T ndarray (t,y,x)
        Each timepoint is a 2D array.
    ims_mask : 2D+T ndarray (t,y,x)
        Each timepoint is a 2D boolean array. True values are pixels to
        be included for analysis.
    ims_corr : 2D+T+RGB ndarray (t,y,x,3)
        Each timepoint is a 2D RGB image where there may be user-supplied
        corrections to the segmentation or mask.
    ims_seeds : 2D+T ndarray (t,y,x)
        Each timepoint is a 2D array with integer region labels.

    Returns
    -------
    ims_labels : 3D numpy array, (t,y,x)
        Each timepoint is a 2D array with integer-labeled regions.
    """
    ims_mask = validate_mask(ims_intensities, ims_mask)

    # Total number of frames
    total_t = np.shape(ims_intensities)[0]

    # Make an (x,y,t) array of time-lapse frames
    ims_labels = np.zeros(np.shape(ims_intensities), dtype=np.uint16)

    # Loop over frames, rescaling each one
    for t in range(total_t):
        if ims_seeds is not None:
            seed = ims_seeds[t]
        else:
            seed = None
        if ims_corr is not None:
            im_intensities_corr, im_mask_corr = apply_corrections(
                ims_intensities[t], ims_corr[t], mask=ims_mask[t]
            )
        else:
            im_intensities_corr = ims_intensities[t]
            im_mask_corr = ims_mask[t]

        ims_labels[t] = epithelium_watershed(
            img_as_ubyte(rescale_intensity(im_intensities_corr)),
            mask=im_mask_corr,
            im_seeds=seed,
        )
    return ims_labels

def segment_epithelium_cellpose_timelapse(ims, cell_diam, tissue_masks):
    ims_lab = np.zeros_like(ims, dtype=np.uint16)
    for t in range(len(ims)):
        print(f"Segmenting frame {t+1} of {len(ims)}")
        ims_lab[t] = segment_epithelium_cellpose(ims[t], cell_diam, tissue_masks[t])
    return(ims_lab)

def largest_object_mask_timelapse(
    ims_intensities, blurring_sigma=15, threshold="adaptive"
):
    """
    Make a mask of the largest bright object in each timelapse timepoint.

    Parameters
    ----------
    ims_intensities : 3D ndarray (t,y,x)
        Each timepoint is a 2D array.
    blurring_sigma: int
        Sigma of Gaussian kernel used to blur the images
    threshold: int or str "adaptive"
        Threshold to separate object from background pixels.
        If "adaptive", Otsu's adaptive thresholding is used.

    Returns
    -------
    ims_mask: 3D ndarray (t,y,x)
        3D boolean array with same shape as ims_intensities. True objects with
        a background of False.
    """
    ims_mask = np.zeros(ims_intensities.shape, dtype=bool)
    for i in range(ims_intensities.shape[0]):
        ims_mask[i] = largest_object_mask(ims_intensities[i], blurring_sigma, threshold)

    return ims_mask
