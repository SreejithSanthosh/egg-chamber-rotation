"""Functions for measuring aspects of a labeled image."""

import numpy as np
from skimage.measure import label, regionprops

def tissue_AP_orientation(tissue_mask):
    """
    Find the orientation of the tissue long axis

    Input
    -----
    tissue_mask: 2d bool array with True at pixels inside tissue

    Output
    -----
    AP_orientation: float, polar. Orientation of the long axis
    of the ellipse approximation of tissue shape. 0 is vertical,
    pi/2 is horizontal.

    """
    tissue_mask_lab = label(tissue_mask)
    props = regionprops(tissue_mask_lab)[0]
    AP_orientation = props['orientation']
    return(AP_orientation)


def tissue_medial_orientation(tissue_mask):
    """
    Find the orientation of the tissue long axis

    Input
    -----
    tissue_mask: 2d bool array with True at pixels inside tissue

    Output
    -----
    medial_orientation: float, polar. Orientation of the long axis
    of the ellipse approximation of tissue shape. 0 is vertical,
    pi/2 is horizontal.

    """

    tissue_mask_lab = label(tissue_mask)
    props = regionprops(tissue_mask_lab)[0]
    medial_orientation = props['orientation'] + np.pi/2
    return(medial_orientation)
