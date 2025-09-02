"""Class for a tracked timelapse."""

import os
import numpy as np
from skimage.measure import regionprops_table, label
from skimage.segmentation import watershed
from imageio.v3 import imread, imwrite
from ..segment import (
    largest_object_mask_timelapse,
    select_border_adjacent,
    segment_epithelium_timelapse,
    segment_epithelium_cellpose_timelapse,
)
from ..utils import validate_mask


class TrackedTimelapse:
    """Base class."""

    # If a new cell has a overlaps at least this much with the previous
    # time point's mask, it is assigned the previous time point's label:
    OVERLAP_MIN = 0.5

    def __init__(
        self, ims_intensities, ims_mask=None, cell_diam=70, basename="timelapse", out_dir=None
    ):
        """
        Initialize TrackedTimelapse object.

        Args
        ----
        ims_intensities : 3D ndarray
            Timelapse with the shape (t, x, y), with integer region labels
        ims_mask : 3D bool ndarray
            Has the same shape as ims
        basename : str
            String to identify all output files
        out_dir : str
            Path to the directory where frames and movies will be saved
        """
        self.ims_intensities = ims_intensities
        self.t_total = np.shape(ims_intensities)[0]
        self.basename = basename
        if out_dir is None:
            self.out_dir = os.getcwd()
        else:
            self.out_dir = out_dir
        self.frames_dir = os.path.join(self.out_dir, basename)
        # Check mask and intialize if none given
        self.ims_mask = validate_mask(self.ims_intensities, ims_mask)
        # Path to a tracked labels file, tissue mask for this dataset
        self.im_path_tracked = os.path.join(
            self.out_dir, f"{self.basename}_tracked.tif"
        )
        self.tissue_mask_path = os.path.join(
            self.out_dir, f"{self.basename}_tissue_mask.tif"
        )

        # Check to see if there is a tracked dataset
        try:
            self.ims_tracked = imread(self.im_path_tracked)
            self.ims_labels = np.copy(self.ims_tracked)
            self.set_mask_from_label_zeros()
            print("Found existing tracked volume.")
        except FileNotFoundError:
            print("Segmenting all timepoints...")

            # Check to see if there is a tissue mask
            try:
                self.tissue_masks = imread(self.tissue_mask_path)
                print("Found existing tissue mask.")
            except FileNotFoundError:
                print("Making tissue mask...")
                self.make_tissue_masks()

            self.segment_all_cellpose(cell_diam=cell_diam, tissue_masks=self.tissue_masks)
            self.ims_tracked = np.copy(self.ims_labels)
            print("Propagating all timepoints...")
            self.propagate_labels(0, self.t_total - 1)
            print("Saving tracked volume...")
            self.save_volume()
            print("    Done.")

    def set_mask_from_label_zeros(self):
        """Set all the zero pixels to be mask."""
        self.ims_mask = self.ims_labels != 0

    def update_mask(self, t, im_labels_mod):
        """Set mask to zero if entire region is set to zero by user."""
        for lab in np.unique(self.ims_labels[t]):
            if np.all(im_labels_mod[self.ims_labels[t] == lab] == 0):
                self.ims_mask[t][self.ims_labels[t] == lab] = 0

    def make_tissue_masks(self):
        """Make masks of the tissue in each frame by blurring, thresholding, finding largest blob"""
        self.tissue_masks = largest_object_mask_timelapse(self.ims_intensities)
        self.tissue_masks_out = self.tissue_masks.astype('uint8')
        self.save_volume(volume="tissue_mask")

    def segment_all(self):
        """Segment the timelapse."""
        self.ims_labels = segment_epithelium_timelapse(
            self.ims_intensities, self.ims_mask
        )

        # Expand the mask to block labeled regions adjacent to image border
        for t in range(self.t_total):
            border_mask = select_border_adjacent(self.ims_labels[t])
            self.ims_mask[t] = self.ims_mask[t] * ~border_mask

        # Apply expanded mask to labeled image series
        self.ims_labels = self.ims_labels * self.ims_mask

    def segment_all_cellpose(self, cell_diam, tissue_masks):
        """Segment the timelapse using cellpose."""
        self.ims_labels = segment_epithelium_cellpose_timelapse(
            self.ims_intensities, cell_diam, tissue_masks
            )

        # Expand the mask to block labeled regions adjacent to image border
        for t in range(self.t_total):
            border_mask = select_border_adjacent(self.ims_labels[t])
            self.ims_mask[t] = self.ims_mask[t] * ~border_mask

        # Apply expanded mask to labeled image series

    def resegment(self, t_start=0, t_stop=None, seeds=None):
        """Resegment the timelapse."""
        self.ims_labels[t_start:t_stop] = segment_epithelium_timelapse(
            self.ims_intensities[t_start:t_stop],
            self.ims_mask[t_start:t_stop],
            ims_seeds=seeds[t_start:t_stop],
        )
        # Put the resegmented labels into ims_tracked
        self.ims_labels = self.ims_labels * self.ims_mask
        self.ims_tracked[t_start:t_stop] = np.copy(self.ims_labels[t_start:t_stop])

    def propagate_labels(self, t_start, t_stop):
        """Propagate labels from t_start to t_stop."""
        if t_stop >= self.t_total:
            t_stop = self.t_total - 1
        for t in range(t_start, t_stop):
            self.propagate_one_timepoint(t)

    def propagate_one_timepoint(self, t):
        """Apply the labels from t to t+1."""
        print(f"Propagating from t={t} to t={t+1}...")
        # Relabel t+1, starting the count at current max + 1
        max_curr_label = np.amax(self.ims_tracked[t])
        self.ims_tracked[t + 1] = self.ims_mask[t + 1] * (
            label(self.ims_tracked[t + 1]) + max_curr_label + 1
        )
        # Get links from t to t+1 and vice versa
        links_curr_to_next = self._get_matches_for_one_t_pair(t, t + 1)
        links_next_to_curr = self._get_matches_for_one_t_pair(t + 1, t)

        # Loop over curr labels that are linked to by at least one next label
        for curr_lab in links_next_to_curr.keys():
            # Get the list of labels in t+1 that link to curr_lab
            next_labs_linking_to_curr = links_next_to_curr[curr_lab]
            # How many cells in t point to the one next label?
            if len(next_labs_linking_to_curr) == 1:
                one_next_lab = next_labs_linking_to_curr[0]
                # Check that the current cell links to at least one next label
                if one_next_lab in links_curr_to_next:
                    curr_labs_linking_to_next = links_curr_to_next[one_next_lab]
                    if len(curr_labs_linking_to_next) <= 1:
                        self.set_label(t + 1, one_next_lab, curr_lab)
                        # label next as curr
                    else:
                        # There is an incorrect merge or lost cell
                        self._resolve_merge(t, one_next_lab, curr_labs_linking_to_next)
            elif len(next_labs_linking_to_curr) > 1:
                # There is an incorrect split or new cell
                self._resolve_split(t, curr_lab, next_labs_linking_to_curr)

    def propagate_one_label(self, t, label):
        """Take a single region label in time t and set the region in t+1."""
        # Find label in t+1
        label_in_next_t = self.get_matching_label(t, label, t + 1)
        # Set label in t+1 to be label in t
        if label_in_next_t != 0:
            self.set_label(t + 1, label_in_next_t, label)

    def get_centroid(self, im):
        """Get centroid of region in boolean 2D image."""
        centroid = regionprops_table(label(im), properties=["centroid"])
        row = centroid["centroid-0"][0]
        col = centroid["centroid-1"][0]
        return int(row), int(col)

    def set_label(self, t, old_label, new_label):
        """Set a label in the tracked dataset."""
        self.ims_tracked[t][self.ims_tracked[t] == old_label] = new_label

    def get_matching_label(self, t_src, lab_src, t_dst):
        """Get the label in the same (r, c) location with different t."""
        row, col = self.get_centroid(self.ims_tracked[t_src] == lab_src)
        lab_dst = self.ims_tracked[t_dst, row, col]
        return lab_dst

    def save_frame(self, t):
        """Save a single tracked frame at the original pixel dimensions."""
        if not os.path.isdir(self.frames_dir):
            os.mkdir(self.frames_dir)
        save_rgb_frame(
            self.ims_intensities[t],
            self.ims_tracked[t],
            self.ims_mask[t],
            im_overlay=overlay_corrections(self.ims_corrections[t]),
            filename=self._make_frame_path(t),
        )

    def save_all_frames(self):
        """Save all tracked and labeled frames."""
        for t in range(self.t_total):
            self.save_frame(t)

    def save_volume(self, volume="tracked", suffix=""):
        """Save a TIF stack of the tracked labels."""
        ims_out = self._pick_volume(volume)
        sep = "" if suffix == "" else "_"
        vol_path = os.path.join(
            self.out_dir, f"{self.basename}_{volume}{sep}{suffix}.tif"
        )
        imwrite(vol_path, ims_out)

    def save_movie(self, volume="tracked", suffix=""):
        """Save an mp4 of the tracked dataset."""
        ims_out = self._pick_volume(volume)
        sep = "" if suffix == "" else "_"
        movie_path = os.path.join(
            self.out_dir, f"{self.basename}_{volume}{sep}{suffix}"
        )
        save_rgb_timelapse(
            self.ims_intensities, ims_out, self.ims_mask, filename=movie_path
        )

    def _pick_volume(self, volume):
        """Select a volume to output."""
        if volume == "tracked":
            ims_out = self.ims_tracked
        elif volume == "tracked_refined":
            ims_out = self.ims_tracked_refined
        elif volume == "tracked_hjs":
            ims_out = self.ims_tracked_hjs
        elif volume == "tissue_mask":
            ims_out = self.tissue_masks_out
        else:
            print(
                'Volume identifier not recognized. Should be "tracked", '
                '"tracked_refined", "tracked_hjs", or "tissue_mask". Saving the "tracked" '
                'volume as the default.'
            )
        return ims_out

    def _get_matches_for_one_t_pair(self, t_src, t_dst):
        """
        Get all matching label pairs from one time point to another.

        Suppose t_src is t0 and t_dst is t1. If a cell splits between
        t0 and t1, then there will be two distinct labels as keys,
        each of which has the same value (a single-element list).
        But if t_src is t1 and t_dst is t0, then there will be a single
        key whose value is a list with two elements.

        Returns: dict, with each key an label in t_dst, and each value is
        a list of labels that link to it.
        """
        src_labels = np.unique(self.ims_tracked[t_src])
        links = {}  # dst_labels are keys

        # Loop over src_labels
        for lab in src_labels:
            if lab != 0:
                match = self.get_matching_label(t_src=t_src, lab_src=lab, t_dst=t_dst)
                if match != 0:
                    if match in links:
                        links[match].append(lab)
                    else:
                        links[match] = [lab]
        return links

    def _resolve_split(self, t, curr_label, ls_of_next_labels):
        """Assess if a possible split is a new cell and then update labels."""
        for next_label in ls_of_next_labels:
            new_cell_bool = self.ims_tracked[t + 1] == next_label
            # Calculate fractional area overlap with previous time point mask
            new_cell_area = np.sum(new_cell_bool)
            overlap_with_curr_mask_area = np.sum(new_cell_bool * self.ims_mask[t])
            overlap_frac = overlap_with_curr_mask_area / new_cell_area
            # If a large enough fractional area of the cells is not masked out,
            # by the current mask, treat as incorrect split
            if overlap_frac > self.OVERLAP_MIN:
                self.set_label(t + 1, next_label, curr_label)

    def _resolve_merge(self, t, next_label, ls_of_curr_labels):
        """Assess if a possible merge is a lost cell and then update labels."""
        overlap_frac_ls = []
        for curr_label in ls_of_curr_labels:
            new_cell_bool = self.ims_tracked[t] == curr_label
            # Calculate fractional area overlap with next time point mask
            new_cell_area = np.sum(new_cell_bool)
            overlap_with_next_mask_area = np.sum(new_cell_bool * self.ims_mask[t + 1])
            overlap_frac = overlap_with_next_mask_area / new_cell_area
            overlap_frac_ls.append(overlap_frac)
        # If all of the current cells have a large enough fractional area
        # not masked out by the next mask, treat as incorrect merge
        if np.all(np.array(overlap_frac_ls) > self.OVERLAP_MIN):
            im_resegmented = _resegment_wrongly_merged_cells(
                im_next=self.ims_intensities[t + 1],
                im_labels_curr=self.ims_tracked[t],
                merged_labels=ls_of_curr_labels,
                mask=self.ims_tracked[t + 1] == next_label,
            )
            # Set the newly segmented labels on the next cell
            self.ims_tracked[t + 1][
                self.ims_tracked[t + 1] == next_label
            ] = im_resegmented[self.ims_tracked[t + 1] == next_label]

    def _make_frame_path(self, t):
        """Make a path for a saved, labels frame."""
        return os.path.join(self.frames_dir, f"{self.basename}_frame_{t:03d}.tif")


def _resegment_wrongly_merged_cells(im_next, im_labels_curr, merged_labels, mask):
    """
    Correct an error in tracking for a single cell at one time point.

    If two cells appear to merge when they shouldn't, this takes their
    positions in the previous timepoint and uses them to generate new
    seeds for a watershed segmentation of the cells in question.

    Parameters
    ----------
    im_next : 2D ndarray, (y,x)
        Next t-step image

    im_labels_curr : 2D ndarray, (y,x)
        One timepoint with integer-labeled regions, one for each cell

    merged_labels : list of integers
        The set of labels in im_labels_curr that has been merged together

    mask : 2D bool ndarray
        True where the merged cell region is

    Returns
    -------
    im_resegmented : 2D ndarray, (y,x)
        Updated current image with integer-labeled regions
    """
    # Array for seeds
    seeds = np.zeros(np.shape(mask))

    for lab in merged_labels:
        # Overlap of the prev labels cell with the curr merge mask
        cell = np.logical_and(im_labels_curr == lab, mask)

        # Use the darkest 10% of pixels in each cell to set the seeds
        cell_masked_array = np.ma.masked_array(im_next, mask=np.invert(cell))
        cell_th = np.quantile(np.ma.compressed(cell_masked_array), np.array((0.1,)))

        seeds[np.ma.filled(cell_masked_array < cell_th, fill_value=0)] = lab

    # Segment using watershed
    im_resegmented = watershed(im_next, markers=seeds, mask=mask)

    return im_resegmented
