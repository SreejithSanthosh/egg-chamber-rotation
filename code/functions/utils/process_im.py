"""Utility image functions."""

import numpy as np

def center_im(im, new_center):
    """
    Center on a point in an image.

    Returns a copy of the image, re-centered on the provided point.
    Fill empty space with zeros if necessary.

    Parameters
    ----------
    im : ndarray
    center : N-element tuple of ints or floats, where N = ndims of im

    Returns
    -------
    im_centered : ndarray
        Same shape as 'im'
    """
    ndims = np.ndim(im)
    im_centered = np.copy(im)
    for dim in range(ndims):
        dim_extent = np.shape(im)[dim]
        dim_center = (dim_extent - 1) / 2
        # Get delta by rounding to closest int
        delta = int(np.round(new_center[dim] - dim_center))
        im_centered = pad_along_1d(im_centered, dim, delta)
        im_centered = trim_along_1d(im_centered, dim, -delta)
    return im_centered


def crop_im(im, cropping_dims):
    """
    Crop an image to provided dimensions.

    Parameters
    ----------
    im : ndarray
    cropping_dims : tuple of ints, length = ndims
        Each element sets the extent along the corresponding dimension

    Returns
    -------
    cropped ndarray with same dimensions as im
    """
    ndims = np.ndim(im)
    slicing_indices = [slice(None)] * ndims
    for dim in range(ndims):
        dim_extent = np.shape(im)[dim]
        final_extent = cropping_dims[dim]
        diff = int((dim_extent - final_extent) / 2)
        slicing_indices[dim] = slice(diff, diff + final_extent)
    return np.copy(im[tuple(slicing_indices)])


def pad_along_1d(arr, dim, delta, fill_val=0):
    """
    Take an arbitrary shaped array and pad along dim to length delta.

    If delta is positive, pad on positive end of dimension 'dim'
    and if delta is negative, pad on negative end of dimension 'dim'.

    Parameters
    ----------
    arr : ndarray
    dim : int
        Dimension index of arr
    delta : int
        Extent of padding
    fill_val : int or float
        Padding value

    Returns
    -------
    ndarray with the same dimensions, padded
    """
    ndims = np.ndim(arr)
    padding_ls = [(0, 0)] * ndims
    if delta >= 0:
        padding_ls[dim] = (0, delta)
    else:
        padding_ls[dim] = (abs(delta), 0)
    return np.pad(arr, tuple(padding_ls), constant_values=fill_val)


def trim_along_1d(arr, dim, delta):
    """
    Take an arbitrary shaped array and crop along dim, removing delta.

    If delta is 3, trim 3 from positive end of dim and if delta is -3,
    trim 3 from negative end of dim.

    Parameters
    ----------
    arr : ndarray
    dim : int
        Dimension index of arr
    delta : int
        Extent of trimming

    Returns
    -------
    ndarray with the same dimensions, trimmed
    """
    ndims = np.ndim(arr)
    slicing_indices = [slice(None)] * ndims
    if delta > 0:
        slicing_indices[dim] = slice(None, -delta)
    else:
        slicing_indices[dim] = slice(abs(delta), None)
    return np.copy(arr[tuple(slicing_indices)])


def trim_zeros_2d(im):
    """
    Trim all rows and columns that are composed entirely of zeros.

    Parameters
    ----------
    im : 2D ndarray

    Returns
    -------
    2D ndarray
    """
    # Trim columns
    im_col_rm = np.delete(im, np.where(~im.any(axis=0))[0], axis=1)
    # Trim rows
    return np.delete(im_col_rm, np.where(~im_col_rm.any(axis=1))[0], axis=0)


def trim_zeros_reference_2d(im_to_trim, im_ref):
    """
    Trim all rows and columns made of zeroes.

    Specifically, find rows and cols in im_ref that are composed entirely
    of zeros, then trim out those corresponding rows and cols in im_to_trim.

    Parameters
    ----------
    im_to_trim: 2D ndarray to be trimmed
    im_ref: 2D ndarray, same dimensions as im_to_trim

    Returns
    -------
    im_trimmed: 2D ndarray
    """
    # Trim columns
    im_ref_col_rm = np.delete(im_ref, np.where(~im_ref.any(axis=0))[0], axis=1)
    im_col_rm = np.delete(im_to_trim, np.where(~im_ref.any(axis=0))[0], axis=1)

    # Trim rows
    im_trimmed = np.delete(im_col_rm, np.where(~im_ref_col_rm.any(axis=1))[0], axis=0)
    return im_trimmed
