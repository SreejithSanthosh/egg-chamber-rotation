"""Functions for handling the interface between two regions."""

import numpy as np
from skimage.filters import sobel
from skimage.graph import route_through_array
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation, binary_erosion
from skimage.segmentation import flood_fill, watershed
import skfmm
from scipy.stats import mode
from ..utils import dilate_simple

def interface_length_segment(cell_a, cell_b):
    """
    Measure straight-line length of an interface.

    Parameters
    ----------
    cell_a, cell_b: 2D bool ndarrays
        Pixels in the cells are True, rest are False

    Returns
    -------
    length : float
        Segment length connecting the interface corners
    """
    e1, e2 = interface_endpoints_coords(cell_a, cell_b)
    length = np.linalg.norm(np.array(e1) - np.array(e2))
    return length

def interface_endpoints_mask(cell_a, cell_b):
    """
    Make a bool mask of the endpoints of an interface between two cells.

    A junction is the edge between two cells. Here the endpoints of the
    junction are defined to be pixels that are immediately outside of the
    two cells. If there are no corners between the cells, returns an
    array of False.

    Parameters
    ----------
    cell_a : 2D bool ndarray
        Pixels in cell are True, rest are False
    cell_b : 2D bool ndarray
        Pixels in cell are True, rest are False

    Returns
    -------
    corners_mask : 2D bool ndarray
        True where corners of two cells are, rest are False
    """
    dilated_a = binary_dilation(cell_a, footprint=np.ones((3, 3)))
    dilated_b = binary_dilation(cell_b, footprint=np.ones((3, 3)))
    edge_interface = np.logical_and(dilated_a, dilated_b)

    # Get outer-most edge of cell pair
    pair = np.logical_or(cell_a, cell_b)
    pair_dilated = binary_dilation(pair)
    pair_edge = np.logical_xor(pair_dilated, pair)

    # Find overlap of the edge masks
    corners_mask = np.logical_and(pair_edge, edge_interface)
    return corners_mask


def interface_endpoints_coords(cell_a, cell_b):
    """
    Find the endpoint coordinates of an interface between two cells.

    See 'interface_endpoints_mask' for how junction endpoints are
    defined. If there are not 2 corners between the cells, raises
    an exception.

    Parameters
    ----------
    cell_a : 2D bool ndarray
        Pixels in the cell are True, rest are False
    cell_b : 2D bool ndarray
        Pixels in the cell are True, rest are False

    Returns
    -------
    endpoints : tuple of tuples
        Stored as ((row1, col1), (row2, col2))
    """
    corners_mask = interface_endpoints_mask(cell_a, cell_b)
    corners_mask = binary_dilation(corners_mask, footprint=np.ones((5, 5)))
    if np.all(~corners_mask):
        raise Exception("Zero endpoints found between these cells")
    # Label the corners and use their centroids as coordinates of the cell interface
    corner_labels = label(corners_mask)
    total = np.max(corner_labels)
    if total == 2:
        centroid_0 = regionprops(corner_labels)[0].centroid
        centroid_1 = regionprops(corner_labels)[1].centroid
        endpoints = (centroid_0, centroid_1)
    else:
        raise Exception(f"Expected 2 corner mask regions; found {total}")

    return endpoints


def interface_shape_edge_method(im, cell_a, cell_b):
    """
    Generate a mask of an interface between two cells.

    Here "interface" is defined as bright blob that spans two neighboring cells.
    It is segmented by doing a watershed with three seeds: (1) The exact mask of
    the cell-cell junction (2 pixels wide), (2 and 3) The darkest 10 percent of
    pixels in each of cell_a and cell_b.

    Parameters
    ----------
    im : 2D ndarray
        Micrograph with cell interface label
    cell_a, cell_b : 2D bool ndarray
        Pixels in each cell are True, rest are False

    Returns
    -------
    interface : 2D bool ndarray
        Pixels are True in the interface, rest are False
    """
    # Use sobel filter to make a mask representing
    # the place where the two masks meet
    edge_a = sobel(cell_a)
    edge_b = sobel(cell_b)
    edge_interface = np.logical_and(edge_a, edge_b) > 0

    # Use the darkest 10% of pixels in each cell to set the seeds
    cell_a_masked_array = np.ma.masked_array(im, mask=np.invert(cell_a))
    cell_b_masked_array = np.ma.masked_array(im, mask=np.invert(cell_b))
    cell_a_th = np.quantile(np.ma.compressed(cell_a_masked_array), np.array((0.1,)))
    cell_b_th = np.quantile(np.ma.compressed(cell_b_masked_array), np.array((0.1,)))

    # Make 3 seeds
    three_seeds = np.copy(edge_interface) * 3
    three_seeds[np.ma.filled(cell_a_masked_array <= cell_a_th, fill_value=0)] = 1
    three_seeds[np.ma.filled(cell_b_masked_array <= cell_b_th, fill_value=0)] = 2

    # Make a joint mask of both neighbors
    joint_mask = np.logical_or(cell_a, cell_b)

    # Take sobel of im
    im_sobel = sobel(im)

    # Segment using watershed
    im_labeled_three_seeds = watershed(im_sobel, markers=three_seeds, mask=joint_mask)

    # Get the label of the object that overlaps most with edge_interface
    interface_label = mode(im_labeled_three_seeds[edge_interface])[0][0]
    interface = im_labeled_three_seeds == interface_label

    return interface


def trim_interface(cell_a, cell_b, interface):
    """
    Trim an interface mask.

    Keep only those pixels whose travel path to them from each sending cell
    is shorter than the travel path from the surrounding regions.

    Parameters
    ----------
    cell_a, cell_b, interface : 2D bool ndarray
        Pixels in the shape are True, rest are False

    Returns
    -------
    interface_updated : 2D bool ndarray
        Pixels in the interface are True, rest are False
    """
    # Make some boolean masks that will be needed
    not_a_nor_b = np.invert(np.logical_or(cell_a, cell_b))
    cell_a_or_b = np.logical_or(cell_a, cell_b)
    cell_a_not_interface = np.logical_and(cell_a, np.invert(interface))
    cell_b_not_interface = np.logical_and(cell_b, np.invert(interface))

    # Travel from cell_b, masking non-interface in cell_a and all other cells
    mask = np.logical_or(cell_a_not_interface, not_a_nor_b)
    phi = np.ma.MaskedArray(np.invert(cell_b), mask)
    dist_in_interface_from_b = skfmm.distance(phi)

    # Travel from cell_a, masking non-interface in cell_a and all other cells
    mask = np.logical_or(cell_b_not_interface, not_a_nor_b)
    phi = np.ma.MaskedArray(np.invert(cell_a), mask)
    dist_in_interface_from_a = skfmm.distance(phi)

    # Travel from background, masking non-interface in cell_a and cell_b
    mask = np.logical_or(cell_a_not_interface, cell_b_not_interface)
    phi = np.ma.MaskedArray(cell_a_or_b, mask)
    dist_in_interface_from_bg = skfmm.distance(phi)

    # Get the updated interface shape
    hemijunction_b_to_a = np.logical_and(
        cell_a, dist_in_interface_from_b < dist_in_interface_from_bg
    )
    hemijunction_a_to_b = np.logical_and(
        cell_b, dist_in_interface_from_a < dist_in_interface_from_bg
    )
    interface_updated = np.logical_or(hemijunction_b_to_a, hemijunction_a_to_b).filled(
        0
    )
    return interface_updated


def refine_junction(cell_a, cell_b, interface):
    """
    Use interface mask to refine a cell-cell junction.

    An "interface" is defined as bright blob that spans two neighboring cells.
    The portion of the interface that overlaps with each cell is called a
    "hemijunction". This function takes the masks of two cells and the mask of
    an interface spanning the two cells, and then updates the two cell shapes so
    that the junction between then takes the shortest curvy path while remaining
    within the interface itself. This is based on the assumption that cell-cell
    interfaces tend to be mostly straight in epithelia.

    Parameters
    ----------
    cell_a, cell_b, interface : 2D bool ndarrays
        Pixels in the shape are True, rest are False

    Returns
    -------
    cell_a_new, cell_b_new : 2D bool ndarrays
        Same input masks, with refined border between them
    """
    # Make a bool image with a junction endpoint set to zero
    im_endpoints = interface_endpoints_mask(cell_a, cell_b)
    im_endpoints_dilated = binary_dilation(im_endpoints, footprint=np.ones((5, 5)))

    # Use the Fast March to calculate distances from the endpoint
    mask = np.invert(interface)
    phi = np.ma.MaskedArray(np.invert(im_endpoints_dilated), mask)
    dist_from_endpoint = skfmm.distance(phi)

    # Make a weight matrix for finding the junction path
    weights = dist_from_endpoint.filled(1000)
    interface_eroded = binary_erosion(interface, footprint=np.ones((3, 3)))
    weights[np.logical_xor(interface_eroded, interface)] += 10

    # Get lowest cost path using endpoint coords as start and end
    e1, e2 = interface_endpoints_coords(cell_a, cell_b)
    indices, cost = route_through_array(
        weights, [int(e1[0]), int(e1[1])], [int(e2[0]), int(e2[1])]
    )
    indices = np.array(indices).T
    path = np.zeros_like(cell_a)
    path[indices[0], indices[1]] = True

    # Make some masks of the cells without the interface
    cell_a_not_interface = np.logical_and(cell_a, np.invert(interface))
    cell_b_not_interface = np.logical_and(cell_b, np.invert(interface))
    cell_a_or_b = np.logical_or(cell_a, cell_b)

    # First fill up from cell a
    seed = (
        np.nonzero(cell_a_not_interface)[0][0],
        np.nonzero(cell_a_not_interface)[1][0],
    )
    flood_ready = np.copy(cell_a_or_b) * 1
    flood_ready[path] = 2
    cell_a_new = flood_fill(flood_ready, seed, 2, connectivity=1) == 2
    cell_b_new = np.logical_xor(cell_a_new, cell_a_or_b)

    # Make masks for the new hemijunctions
    hj_a = np.logical_and(cell_a_new, interface)
    hj_b = np.logical_xor(interface, hj_a)

    # If cell a's hemijunction is smaller, put the path on other side
    if np.sum(hj_a) < np.sum(hj_b):
        hj_a = np.logical_xor(path, hj_a)
        hj_b = np.logical_xor(interface, hj_a)
        cell_a_new = np.logical_or(hj_a, cell_a_not_interface)
        cell_b_new = np.logical_or(hj_b, cell_b_not_interface)

    return cell_a_new, cell_b_new


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