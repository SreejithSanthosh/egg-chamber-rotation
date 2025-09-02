"""Utility functions for using napari."""

import numpy as np
from skimage.measure import label, regionprops_table


def select_layer(viewer, layer):
    """Select the layer and deselect all the rest."""
    indices = list(range(len(viewer.layers)))
    layer_names = [viewer.layers[i].name for i in indices]
    index_of_layer_to_select = layer_names.index(layer.name)
    # Select the labels layer
    for i in indices:
        viewer.layers.selection.add(viewer.layers[i])
    indices.remove(index_of_layer_to_select)
    for j in indices:
        viewer.layers.selection.remove(viewer.layers[j])
    return


def make_centroid_points_layer(viewer, labels_layer):
    """
    Put text labels on each region by replacing points layer.

    Parameters
    ----------
    viewer : napari Viewer object
    labels_layer : napari Layer object
        Should be regions with distinct integer labels.

    Returns
    -------
    points_layer : napari Layer object
        One point for each region, placed at its centroid, with a text label
        that matches the label of the region.
    """
    im = np.copy(labels_layer.data)
    point_ls = []
    label_ls = []
    if np.ndim(im) == 2:
        im = np.expand_dims(im, axis=0)
    for t in range(np.shape(im)[0]):
        props = regionprops_table(im[t], properties=("label", "centroid"))
        frame_points = np.vstack(
            (
                np.ones_like(props["centroid-0"]) * t,
                props["centroid-0"],
                props["centroid-1"],
            )
        ).T
        point_ls.append(frame_points)
        label_ls.append(props["label"])
    all_points = np.vstack(point_ls)
    all_labels = np.hstack(label_ls)

    # Make a string to be a name for the points layer
    points_layer_name = labels_layer.name + "_text"
    layer_names = [viewer.layers[i].name for i in range(len(viewer.layers))]

    # See if there is an old points layer, and if so, remove it
    if points_layer_name in layer_names:
        viewer.layers.pop(layer_names.index(points_layer_name))

    # Make new points layer
    points_layer = make_points_layer(viewer, all_points, all_labels, points_layer_name)
    points_layer.size = 0
    return points_layer


def make_points_layer(viewer, points, text="", layer_name="points"):
    """
    Make a points layer.

    Parameters
    ----------
    viewer : napari Viewer object
    points : ndarray
        axis 0 is number of dimensions
        axis 1 is the number of
        points n is the number of points.
    text : str or list with length n or ndarray with shape (n,)
    layer_name : str

    Returns
    -------
    points_layer : napari Layer object
    """
    if isinstance(text, str):
        text_array = np.array([text] * points[:, 0].shape[0])
    else:
        text_array = np.array(text)
    if text_array.shape != points[:, 0].shape:
        raise ValueError(
            'If "text" is a str, it can be any length. If "text" is a list, '
            "it should have length n, where n matches the size of the second "
            'axis of "points". If "text" is an array, it must be shape (n,)."'
        )
    text_params = {
        "text": "{text_label}",
        "size": 7,
        "color": "white",
        "anchor": "center",
        "translation": np.array((-3, 0)),
    }
    # Make new points layer
    points_layer = viewer.add_points(
        points,
        size=1.0,
        properties={"text_label": text_array},
        text=text_params,
        name=layer_name,
    )
    return points_layer


def resegment_tt_in_viewer(viewer, layer, tt, t_steps_to_resegment):
    """Resegment some number of t_steps."""
    # Determine the final t-step to resegment
    t_curr = int(viewer.cursor.position[0])
    if t_curr + t_steps_to_resegment - 1 > tt.t_total:
        t_reseg_max = tt.t_total - 1
    else:
        t_reseg_max = t_curr + t_steps_to_resegment - 1
    msg = f"Resegmenting, beginning at t={t_curr}..."
    viewer.status = msg
    print(msg)
    # Get current label info as a stack
    im = np.copy(layer.data)
    tt.update_mask(t_curr, im[t_curr])
    # Resegment and propagate labels
    tt.resegment(t_curr, t_reseg_max, seeds=im)
    # Propagate labels if t_curr isn't final timepoint
    if t_curr < tt.t_total - 1:
        tt.propagate_labels(t_curr, t_reseg_max)
    # Set resegmented data to the layer
    layer.data = tt.ims_tracked
    # Replace the points layer and then select it
    make_centroid_points_layer(viewer, layer)
    select_layer(viewer, layer)
    print("Saving a TIF of the current tracked volume...")
    tt.save_volume(volume="tracked", suffix="tmp")
    print("    Done.")
    return


def count_label_lifetimes(ims_labels):
    """Make a list of labels and a matching list of label lifetimes."""
    instances = []
    for i in range(ims_labels.shape[0]):
        unique_labels = np.unique(ims_labels[i])
        instances.append(unique_labels)
    lifespans = np.bincount(np.concatenate(instances))
    label_ids = np.arange(lifespans.shape[0])
    # Get indices needed to reverse sort the lifespans
    lifespans_nz = lifespans[np.nonzero(lifespans)]
    label_ids_nz = label_ids[np.nonzero(lifespans)]
    # sort_indices = np.flip(np.argsort(lifespans_nz))
    # ls_lifespans = lifespans_nz[sort_indices].tolist()
    # ls_label_ids = label_ids_nz[sort_indices].tolist()
    ls_lifespans = lifespans_nz.tolist()
    ls_label_ids = label_ids_nz.tolist()
    return ls_label_ids, ls_lifespans


def flag_discontinuous_labels(ims_labels):
    """Generate list of strings flagging discontinuous regions."""
    data_out = []
    for t in range(ims_labels.shape[0]):
        labs = np.unique(ims_labels[t])
        for lab in labs:
            n = np.max(label(ims_labels[t] == lab, connectivity=1))
            if n > 1:
                data_out.append(np.array((t, lab, n)))
    if len(data_out) > 0:
        return np.vstack(data_out)
    else:
        return None
