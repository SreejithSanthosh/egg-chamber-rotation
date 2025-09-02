"""Utility functions for image processing."""

import numpy as np


def validate_mask(im, mask):
    """
    Check dtype of mask; if 'None', make a blank one.

    Parameters
    ----------
    mask : 'None' or ndarray
        The input mask.

    Returns
    -------
    mask : bool ndarray
        If 'None' was given for the mask, it is a volume of all 1s.

    Raises
    ------
    TypeError
        If the given array is not a dtype('bool').
    """
    # Check type of mask; if absent, make a blank one
    if mask is not None:
        if mask.dtype != np.dtype("bool"):
            raise TypeError("Mask needs to be dtype('bool')")
        else:
            mask_out = np.copy(mask)
    else:
        mask_out = np.ones(np.shape(im), dtype="bool")

    return mask_out
