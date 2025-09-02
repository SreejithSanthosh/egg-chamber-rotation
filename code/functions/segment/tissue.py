"""
Segmentation functions that operate on 2D numpy array representations.

Designed for working with images of biological tissues.
"""
from cellpose import models, utils, io, plot
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, disk, binary_dilation, binary_erosion
from skimage.segmentation import clear_border, flood_fill, watershed
from ..utils import validate_mask, dilate_simple, pol_to_cart
from .interface import (
    edge_between_neighbors,
    interface_endpoints_mask,
    interface_shape_edge_method,
    refine_junction,
    trim_interface,
)

from imageio.v3 import imwrite #TEMP

def edge_between_neighbors(cell_a, cell_b):
    """
    Make 2D bool array for edge between neighbors.

    Parameters
    ----------
    cell_a, cell_b : 2D bool ndarrays
        Pixels in the sending cell are True, rest are False

    Returns
    -------
    edge : 2D bool ndarrays
        2-pixel wide mask formed by dilating both cells
    """
    edge = np.logical_and(dilate_simple(cell_a), dilate_simple(cell_b))
    return edge


def epithelium_watershed(
    im,
    mask=None,
    im_seeds=None,
    blurring_sigma=3,
    threshold_sigma=51,
    erosions=3,
    min_obj_size=100,
    make_background_seed=False,
    background_seed_dilations=0,
):
    """
    Watershed an epithelium.

    Take a 2D micrograph of an epithelium and segment it into labeled
    cell regions. Expects an input image in which connected regions
    of relatively dark pixels are separated by relatively lighter
    pixels.

    If im_seeds is supplied, loop over the new seeds (im_labeled_centers),
    and if one of them overlaps enough with a label in im_seeds, erode the
    labeled region and then copy it into im_labeled_centers before running the
    watershed.

    Parameters
    ----------
    im : 2D ndarray
        Micrograph with cell interface label
    mask : 2D bool ndarray, same shape as im
        True pixels are kept, False pixels are masked
    im_weights : 2D ndarray, same shape as im
        Segmentation is done on im * im_weights
        An array of 1s leaves im unchanged
    blurring_sigma : int
        Sigma of Gaussian kernel used to blur the image
    threshold_sigma : int
        Sigma of Gaussian for locally adaptive threshold function
    erosions : int
        Number of erosions to perform when separating region centers
    min_obj_size : int
        Objects with an area smaller than this threshold are removed
    make_background_seed : bool
        Whether to expand mask and then floodfill background to make a
        unitary background seed for watershed.
    background_seed_dilations : int
        How many dilations to apply to mask before floodfilling background

    Returns
    -------
    im_labeled_regions : 2D ndarray
        Each object has a unique integer ID
    """
    mask = validate_mask(im, mask)
    # Gaussian blur
    im_blurred = gaussian(im, sigma=blurring_sigma, preserve_range=True)
    # Adaptive threshold, inverting image
    adap_th = threshold_local(im_blurred, block_size=threshold_sigma)
    im_thresholded = im_blurred < adap_th
    # Dilate mask
    if make_background_seed:
        for _ in range(background_seed_dilations):
            mask = dilate_simple(mask)
    # Set masked pixels to zero
    im_thresholded[mask == 0] = 0
    # Fill holes if no background seed needed
    if make_background_seed:
        im_ready_to_erode = flood_fill(im_thresholded, (0, 0), True)
    else:
        im_ready_to_erode = binary_fill_holes(im_thresholded)
    # Erode objects
    im_eroded = np.copy(im_ready_to_erode)
    for _ in range(erosions):
        im_eroded = binary_erosion(im_eroded)
    # Remove small objects
    im_seg = remove_small_objects(im_eroded, min_size=min_obj_size)
    # Label regions
    im_labeled_centers = label(im_seg)

    # Incorporate im_seeds into im_labeled_centers before watershed
    if im_seeds is not None:
        for lab in np.unique(im_labeled_centers):
            seed_region = im_seeds[im_labeled_centers == lab]
            if np.any(seed_region == 0):
                im_labeled_centers[im_labeled_centers == lab] = 0
        im_labeled_centers[im_seeds != 0] = im_seeds[im_seeds != 0]

    # Watershed segmentation using the labeled centers as seeds
    im_labeled_regions = watershed(im_blurred, im_labeled_centers, mask=mask)
    return im_labeled_regions


def largest_object_mask(im, blurring_sigma=15, threshold="adaptive"):
    """
    Make a mask of the largest bright object in an image.

    Make a mask containing the largest bright region of an image following
    Gaussian blurring to remove small-scale variation. Bright object is True,
    other regions False. Accepts optional blurring sigma and threshold value
    arguments, or else uses default blurring_sigma and adaptive thresholding.

    Parameters
    ----------
    im: 2D ndarray
        Grayscale image to be masked with bright features, dark background
    blurring_sigma: int
        Sigma of Gaussian kernel used to blur the image
    threshold: int or str "adaptive"
        Threshold to separate object from background pixels.
        If "adaptive", Otsu's adaptive thresholding is used.

    Returns
    -------
    mask: 2D bool ndarray
        Same shape as im. True where largest bright object was identified,
        False elsewhere
    """
    im_blurred = gaussian(im, sigma=blurring_sigma, preserve_range=True)
    if threshold == "adaptive":
        threshold = threshold_otsu(im_blurred)
    im_thresholded = im_blurred > threshold

    if np.amax(im_thresholded) == False:
        raise ValueError("All image intensities are below the threshold")
    else:
        im_labeled_regions = label(im_thresholded)
        mask_with_holes = (
            im_labeled_regions
            == np.argmax(np.bincount(im_labeled_regions.flat)[1:]) + 1
        )
        mask = binary_fill_holes(mask_with_holes)
    return mask


def neighbor_array_nr(im, mask=None, periphery_excluded=True):
    """
    Make an non-redundant array of neighbor region pairs.

    Take a 2D ndarray with regions labeled by integers, and return a
    list of two element lists. First element is an integer label of a
    a region. Second element is an array with shape (N,) where N is the
    number of regions neighboring the first element label. The array
    stores the set of neighbor labels.

    Parameters
    ----------
    im : 2D ndarray
        Labeled image with unique integers for every region
    mask : 2D bool ndarray
        True pixels are kept, False pixels are masked
    periphery_excluded : bool

    Returns
    -------
    neighbor_array : TODO finish
    """
    mask = validate_mask(im, mask)
    # Increment all the labels, to make sure there is no zero
    # Zeros will be reserved for masked pixels
    im2 = np.copy(im) + np.ones(np.shape(im), dtype="uint16")

    # Set masked pixels to zero
    im2[mask == False] = 0

    # Determine the of region IDs to be returned
    unique_labels = np.unique(im2)
    if periphery_excluded:
        unique_labels_in_field = np.unique(im2 * select_in_field(im2, mask))
    else:
        unique_labels_in_field = np.copy(unique_labels)

    # Iterate over labels, appending to a list of pairs
    neighbor_list = []
    for id in list(unique_labels):
        if id != 0 and id in unique_labels_in_field:
            region = im2 == id
            dilated = dilate_simple(region)
            neighbors_plus_self = set(np.unique(np.extract(dilated, im2)) - 1)
            neighbors = neighbors_plus_self - set([id - 1])
            # Make a (2,n) array of this id and its neighbor ids
            a = np.array(list(neighbors))
            b = np.full_like(a, id - 1)
            neighbor_list.append(np.vstack((b, a)).T)

    # Redundant array of all neighbor pairs
    neighbor_array = np.vstack(tuple(neighbor_list))

    # Remove duplicates by keeping cases where first is greater than second
    keepers = neighbor_array[:, 0] > neighbor_array[:, 1]
    neighbor_array = neighbor_array[keepers]
    return neighbor_array


def select_border_adjacent(im):
    """
    Select regions of image that are adjacent to image border.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values

    Returns
    -------
    border_adjacent : bool ndarray
        True where regions are adjacent to border
    """
    border_adjacent = clear_border(label(im)) == 0
    return border_adjacent


def select_in_field(im, mask=None):
    """
    Select regions that are adjacent to neither border nor mask.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    mask : bool ndarray
        Optional mask, same shape as im

    Returns
    -------
    in_field : bool ndarray
        True where regions are with True part of mask, and are
        not adjacent to mask edge nor image border
    """
    mask = validate_mask(im, mask)

    # Make the masks that will be combined
    mask_adjacent = select_mask_adjacent(im, mask)
    masked_or_mask_adjacent = np.logical_or(mask_adjacent, np.invert(mask))
    border_adjacent = select_border_adjacent(im)

    # Combine and invert the masks
    excluded = np.logical_or(masked_or_mask_adjacent, border_adjacent)
    in_field = np.invert(excluded)

    return in_field


def select_mask_adjacent(im, mask=None):
    """
    Select regions of image that are adjacent to a mask.

    Parameters
    ----------
    im : ndarray
        Regions labeled with unique values
    mask : bool ndarray
        Optional mask, same shape as im

    Returns
    -------
    mask_adjacent : bool ndarray
        True where regions within mask are adjacent to mask;
        returns all False if no mask is provided
    """
    if mask is None or np.all(mask):
        return np.zeros(np.shape(im), dtype=bool)

    # Apply mask, then relabel so that labels count from 1 sequentially
    im_masked = np.copy(im) * mask
    im_labels = label(im_masked)
    regions = np.unique(im_labels)
    mask_eroded = binary_erosion(mask)

    # Get IDs in True part of mask adjacent to False part of mask
    peripheral_ids = np.unique(np.invert(mask_eroded) * im_labels)

    # Make bool array of same length as regions, True where
    # region ID are adjacent to the mask
    peripheral_bools = np.isin(regions, peripheral_ids)

    # Apply bool array to labeled image to mask final mask
    mask_adjacent = peripheral_bools[im_labels] * mask

    return mask_adjacent


def tissue_axis_mask(tissue_mask, region_width, axis_orientation, centroid=None):
    """
    Make a mask of the region within a tissue and along an axis.

    Notes:
        Works best if the tissue does not extend to image boundaries,
        especially image corners. Automatic centroid finding will be
        inaccurate if the tissue extends beyond image boundaries.
        Regions near image corners are excluded from the mask
        in some contexts.

    Inputs
    -----
    tissue_mask: 2D bool (r,c) array with
        True pixels inside tissue, False background
    region_width: int, in px.
        The width of the masked region in the direction orthogonal
        to the axis.
    axis_orientation: float, polar. Axis orientation of axis_mask.
        0 is vertical, pi/2 horizontal.
    centroid: None or tuple of ints (r,c). Point through which the
        axis passes. By default, the centroid of tissue_mask is used.

    Output
    -----
    axis_mask: 2d bool array with same shape as tissue_mask. Pixels
        near the axis line are True.

    """

    # Find centroid, orientation
    tissue_mask_lab = label(tissue_mask)
    props = regionprops(tissue_mask_lab)[0]
    if centroid == None:
        r_centroid, c_centroid = props.centroid
    else:
        r_centroid = centroid[0]
        c_centroid = centroid[1]
    m_r, m_c = pol_to_cart(axis_orientation)

    # Calculate axis mask-bounding points
    # Horizontal line case
    if m_r > -0.01 and m_r < 0.01:
        r1 = r_centroid
        c1 = 0
        r2 = r_centroid
        c2 = len(tissue_mask[0,:]-1)

        r3 = r_centroid - region_width/2
        c3 = 0
        r4 = r_centroid - region_width/2
        c4 = len(tissue_mask[0,:])

        r5 = r_centroid + region_width/2
        c5 = 0
        r6 = r_centroid + region_width/2
        c6 = len(tissue_mask[0,:])

        if r3 < 0:
            r3 = 0
            r4 = 0
        if r5 > len(tissue_mask)-1:
            r5 = len(tissue_mask)-1
            r6 = len(tissue_mask)-1

    elif m_c > -0.01 and m_c < 0.01:
        r1 = 0
        c1 = c_centroid
        r2 = len(tissue_mask)
        c2 = c_centroid

        r3 = 0
        c3 = c_centroid - region_width/2
        r4 = len(tissue_mask)
        c4 = c_centroid - region_width/2

        r5 = 0
        c5 = c_centroid + region_width/2
        r6 = len(tissue_mask)
        c6 = c_centroid + region_width/2

        if c3 < 0:
            c3 = 0
            c4 = 0
        if c5 > len(tissue_mask[0,:])-1:
            c5 = len(tissue_mask[0,:])-1
            c6 = len(tissue_mask[0,:])-1

    # Not horizontal, not vertical
    else:
        m = m_r/m_c
        c1 = 0
        c2 = len(tissue_mask[0,:])
        r1 = m * (c1-c_centroid) + r_centroid
        r2 = m * (c2-c_centroid) + r_centroid

        # Shift the medial line in each direction along the AP axis
        n = n = np.sqrt((c2-c1)**2 + (r2-r1)**2)
        dc = 0.5*region_width/n * (r1-r2)
        dr = 0.5*region_width/n * (c2-c1)
        c3 = c1 + dc
        c4 = c2 + dc
        r3 = r1 + dr
        r4 = r2 + dr
        c5 = c1 - dc
        c6 = c2 - dc
        r5 = r1 - dr
        r6 = r2 - dr

        # Make sure bounding points are in the array
        c3_list = []
        r3_list = []
        c5_list = []
        r5_list = []
        for r in range(len(tissue_mask)):
            c = (r - r3) / m + c3
            if c >=0 and c<len(tissue_mask[0,:]):
                c3_list.append(c)
                r3_list.append(r)
            c = (r - r5) / m + c5
            if c >=0 and c<len(tissue_mask[0,:]):
                c5_list.append(c)
                r5_list.append(r)

        c3 = min(c3_list)
        c3_ind = c3_list.index(c3)
        c4 = max(c3_list)
        c4_ind = c3_list.index(c4)
        r3 = r3_list[c3_ind]
        r4 = r3_list[c4_ind]

        c5 = min(c5_list)
        c5_ind = c5_list.index(c5)
        c6 = max(c5_list)
        c6_ind = c5_list.index(c6)
        r5 = r5_list[c5_ind]
        r6 = r5_list[c6_ind]

    # Turn corner points to integers (needed for making polygon mask), keep within image
    r_list = [r3,r4,r6,r5]
    c_list = [c3,c4,c6,c5]
    r_list = [int(n) for n in r_list]
    c_list = [int(n) for n in c_list]
    r_list = [len(tissue_mask)-1 if n >= len(tissue_mask) else n for n in r_list]
    c_list = [len(tissue_mask[0,:])-1 if n >= len(tissue_mask[0,:]) else n for n in c_list]

    # Make axis mask
    axis_mask_untrimmed = np.zeros_like(tissue_mask, dtype=bool)
    rr,cc = polygon(r_list,c_list)
    axis_mask_untrimmed[rr,cc] = True
    axis_mask = axis_mask_untrimmed * tissue_mask

    return(axis_mask)


def segment_epithelium_cellpose(im, cell_diam, tissue_mask):
    """
    Segment the cells in an image using cellpose.

    Parameters
    ------
    im: 2d array, intensities image
    cell_diam: int, px. approximate width of cells in the image.
    tissue_mask: 2d bool array, same shape as im, tissue==True

    Returns
    ------
    im_lab_16bit: 2d uint16 array with same shape as im. Labeled cells.
    """
    # Define model parameters, run the model
    model = models.Cellpose(gpu=False, model_type='cyto')
    im_lab, flows, style, diam = model.eval(im, diameter=cell_diam,channels=[0,0])

    # Fill gaps between cells
    tissue_mask_eroded = binary_erosion(tissue_mask, footprint=disk(int(cell_diam)))
    tissue_mask_joint = np.logical_or(im_lab, tissue_mask_eroded)
    tissue_mask_filled = binary_fill_holes(tissue_mask_joint)

    # Remove cells that do not contact the tissue mask
    tissue_labs = label(tissue_mask_filled)
    tissue_mask_contiguous = np.zeros_like(tissue_mask_filled, dtype=bool)
    tissue_regions = regionprops(tissue_labs)
    for region in tissue_regions:
        lab = region.label
        region_mask = tissue_labs == lab
        if np.any(np.logical_and(tissue_mask, region_mask)):
            tissue_mask_contiguous[region_mask] = True        
            
    # Fill gaps between cells within the expanded, contiguous tissue_mask
    im_lab_filled = watershed(im, markers=im_lab, mask=tissue_mask_filled)

    # Remove cells not contacting the expanded, contiguous tissue_mask
    im_lab_contiguous = np.copy(im_lab_filled)
    im_lab_contiguous[~tissue_mask_contiguous] = 0

    # Reformat
    im_lab_16bit = im_lab_contiguous.astype(np.uint16)
    return(im_lab_16bit)


def cell_edges_mask(im, edge_dilation_factor, mask=None, periphery_excluded=True):
    """
    Make a bool mask of all edge regions between segmented cells.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values. 0 regions are treated as
        background, masked out.
    edge_dilation_factor: int
        Radius of the disk-shaped structuring element by which the edges
        will be dilated (in px)
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether edges of cells touching the image or mask border
        should be included in the returned mask

    Returns
    -------
    edges_mask : 2D bool ndarray
        True where dilated cell edges are, elsewhere False
    """
    # Make mask of region to be included
    mask = validate_mask(im, mask)
    mask = mask * (im > 1)
    if periphery_excluded is True:
        mask = select_in_field(im, mask)
    im_inbounds = im * mask
    # Make array of cell neighbor pairs (non-redundant)
    neighbor_pairs_raw = neighbor_array_nr(im_inbounds)
    neighbor_pairs = neighbor_pairs_raw[neighbor_pairs_raw[:, 1] > 0]
    # Make structuring element for edge dilation
    edge_dil_shape = disk(edge_dilation_factor)
    # Looping through all neighbor pairs, find edges, add to edge mask
    edges_mask = np.zeros_like(im, dtype=bool)
    for i in range(len(neighbor_pairs)):
        cell_a = im == neighbor_pairs[i][0]
        cell_b = im == neighbor_pairs[i][1]
        edge = edge_between_neighbors(cell_a, cell_b)
        edge_dil = binary_dilation(edge, footprint=edge_dil_shape)
        edges_mask[edge_dil] = True
    return edges_mask


def cell_interiors_mask(im, edge_dilation_factor, mask=None, periphery_excluded=True):
    """
    Make a bool mask of non-edge regions of segmented cells.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values. 0 regions are treated as
        background, masked out.
    edge_dilation_factor: int
        Radius of the disk-shaped structuring element by which the edges
        will be dilated (in px)
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether interiors of cells touching the image or mask border
        should be included in the returned mask

    Returns
    -------
    interiors_mask : 2D bool ndarray
        True in non-cell-edge regions, elsewhere false
    """
    # Make structuring element for edge dilation
    edge_dil_shape = disk(edge_dilation_factor)
    # Make mask of region to be included
    mask = validate_mask(im, mask)
    mask = mask * (im > 1)
    if periphery_excluded is True:
        mask = select_in_field(im, mask)
    # Remove edges at periphery
    mask = binary_erosion(mask, footprint=edge_dil_shape)
    im_inbounds = im * mask
    # Make array of cell neighbor pairs (non-redundant)
    neighbor_pairs_raw = neighbor_array_nr(im_inbounds)
    neighbor_pairs = neighbor_pairs_raw[neighbor_pairs_raw[:, 1] > 0]
    # Loop through neighbor pairs, find edges, remove from interiors_mask
    interiors_mask = im_inbounds > 0
    for i in range(len(neighbor_pairs)):
        cell_a = im == neighbor_pairs[i][0]
        cell_b = im == neighbor_pairs[i][1]
        edge = edge_between_neighbors(cell_a, cell_b)
        edge_dil = binary_dilation(edge, footprint=edge_dil_shape)
        interiors_mask[edge_dil] = False
    return interiors_mask


def cell_vertices_mask(im, vertex_dilation_factor, mask=None, periphery_excluded=True):
    """
    Make a bool mask of all vertex regions of segmented cells.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values. 0 regions are treated as
        background, masked out.
    vertex_dilation_factor: int
        Radius of the disk-shaped structuring element by which the vertices
        will be dilated (in px)
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether vertices of the regions touching the image or mask border
        should be included in the returned mask

    Returns
    -------
    vertex_mask_dil : 2D bool ndarray
        True where dilated cell vertices are, elsewhere False
    """
    # Make mask of region to be included
    mask = validate_mask(im, mask)
    mask = mask * (im > 1)
    if periphery_excluded is True:
        mask = select_in_field(im, mask)
    im_inbounds = im * mask
    # Make array of cell neighbor pairs (non-redundant)
    neighbor_pairs_raw = neighbor_array_nr(im_inbounds)
    neighbor_pairs = neighbor_pairs_raw[neighbor_pairs_raw[:, 1] > 0]
    # Loop through neighbor pairs, find interface endpoints,
    # add to vertex_mask
    vertex_mask = np.zeros_like(im)
    for i in range(len(neighbor_pairs)):
        cell_a = im == neighbor_pairs[i][0]
        cell_b = im == neighbor_pairs[i][1]
        vertices = interface_endpoints_mask(cell_a, cell_b)
        vertex_mask[vertices] = True
    # Dilate the vertices
    vertex_dil_shape = disk(vertex_dilation_factor)
    vertex_mask_dil = binary_dilation(vertex_mask, footprint=vertex_dil_shape)
    return vertex_mask_dil
