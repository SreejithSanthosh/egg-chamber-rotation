"""Functions for making video output from timelapses."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.measure import regionprops
from ..utils import validate_mask
from ..segment import select_in_field
from .overlay_elements import overlay_random_colors


def save_rgb_timelapse(
    ims_raw,
    ims_labeled,
    ims_mask=None,
    periphery_excluded=False,
    alpha=0.3,
    fps=8,
    dpi=100,
    filename="labeled_regions",
):
    """
    Save an mp4 of the timelapse in the working directory.

    The raw micrograph channel (ims_raw) is shown in grayscale, and overlayed
    on top are colors determined by im_labeled. The region labels are
    also shown in text. Frame numbers are displayed in the corner.

    Parameters
    ----------
    ims_raw : 2D ndarray (N, M)
        Shown as grayscale intensities
    ims_labeled : 2D ndarray (N, M)
        Regions labeled with unique values
    ims_mask : bool ndarray, optional
        Optional mask, same shape as im_raw and im_labeled
    periphery_excluded : bool, optional
        Whether regions touching the border or mask
        should be included in the exported movie
    alpha : float, optional
        Transparency from 0 to 1
    fps : int, optional
        Frames per second of the final image
    dpi : int, optional
        Resolution of final movie in dots per inch
    filename : str, optional
        Base name for saved file, without extension

    Returns
    -------
    nothing
    """
    # Calculate the interval between frames in milliseconds
    interval = (1 / fps) * 1000

    # Calculate output size in inches
    t_total, height_px, width_px = np.shape(ims_raw)
    height_inches = height_px / dpi
    width_inches = width_px / dpi

    # Check mask, make a blank one if none provided, then apply to image
    ims_mask = validate_mask(ims_raw, ims_mask)
    ims_labeled_masked = np.copy(ims_labeled) * ims_mask

    # Remove peripheral regions if periphery_excluded is True
    if periphery_excluded:
        ims_labeled_masked = ims_labeled_masked * select_in_field(
            ims_labeled_masked, ims_mask
        )

    # Set up a matplotlib Figure with Axes
    fig = plt.figure(figsize=(width_inches, height_inches))
    ax = fig.add_axes([0, 0, 1, 1])

    # List to store the matplotlib Artists for each frame
    frame_artists = []

    # Loop over t, appending Artists
    for t in range(t_total):
        # Artist for raw image
        art_im_raw = ax.imshow(ims_raw[t], cmap=plt.cm.gray)

        # Artist for color overlay
        art_im_colors = ax.imshow(
            overlay_random_colors(
                ims_labeled_masked[t],
                ims_mask[t],
                periphery_excluded=False,
                alpha=alpha,
            )
        )

        # Get properties of the labeled regions
        centroid_list = []
        for region in regionprops(ims_labeled_masked[t]):
            centroid_row, centroid_col = region.centroid
            # Store centroids as (x,y) coordinates
            centroid_list.append(np.array((centroid_col, centroid_row)))

        # Loop over regions, generating a text label for each one
        frame_text_artists = []
        for i in range(len(centroid_list)):
            # Convert centroids to integers in order to index into image
            x = int(centroid_list[i][0])
            y = int(centroid_list[i][1])
            # Get string label from integer coordinates
            s = str(ims_labeled[t][y][x])
            # Place region label text and store matplotlib Artist
            artist = ax.text(
                x,
                y,
                s,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="white",
            )
            frame_text_artists.append(artist)

        # Add a frame marker in the corner
        artist = ax.text(
            20,
            20,
            "frame = " + str(t),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=14,
            color="white",
        )
        frame_text_artists.append(artist)
        frame_artists.append([art_im_raw, art_im_colors] + frame_text_artists)

    # Adjust plot
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # Make animation
    im_ani = animation.ArtistAnimation(fig, frame_artists, interval=interval, blit=True)
    # Save animation
    im_ani.save(f"{filename}.mp4", dpi=dpi)


def save_rgb_frame(
    im_raw,
    im_labeled,
    im_mask=None,
    im_overlay=None,
    periphery_excluded=False,
    alpha=0.3,
    dpi=100,
    filename="labeled_frame.tif",
):
    """
    Save a frame of a labeled cell in the working directory.

    The raw micrograph channel (im_raw) is shown in grayscale, and overlayed
    on top are colors determined by im_labeled. The region labels are
    also shown in text. Frame numbers are displayed in the corner.

    Parameters
    ----------
    im_raw : 2D ndarray (y, x)
        Shown as grayscale intensities
    im_labeled : 2D ndarray (y, x)
        Regions labeled with unique values
    im_mask : bool ndarray
        Optional mask, same shape as im_raw and im_labeled
    im_overlay : 3D ndarray (y, x, 4)
        Optional overlay RGBA image
    periphery_excluded : bool
        Whether regions touching the border or mask
        should be included in the exported movie
    alpha : float
        Transparency from 0 to 1
    dpi : int
        Resolution of final movie in dots per inch
    filename : str
        Base name for saved file, without extension

    Returns
    -------
    nothing
    """
    # Check mask, make a blank one if none provided, then apply to image
    im_mask = validate_mask(im_raw, im_mask)
    im_labeled_masked = np.copy(im_labeled) * im_mask

    # Remove peripheral regions if periphery_excluded is True
    if periphery_excluded:
        im_labeled_masked = im_labeled_masked * select_in_field(
            im_labeled_masked, im_mask
        )

    # Set up the Figure and Axes
    height_px, width_px = np.shape(im_labeled)
    height_inches = height_px / dpi
    width_inches = width_px / dpi
    fig = plt.figure(figsize=(width_inches, height_inches))
    ax = fig.add_axes([0, 0, 1, 1])

    # Plot the intensites and color overlays
    ax.imshow(im_raw, cmap=plt.cm.gray)
    ax.imshow(overlay_random_colors(im_labeled, alpha=alpha))

    # Get properties of the labeled regions
    centroid_list = []
    for region in regionprops(im_labeled_masked):
        centroid_row, centroid_col = region.centroid
        # Store centroids as (x,y) coordinates
        centroid_list.append(np.array((centroid_col, centroid_row)))

    # Loop over regions, generating a text label for each one
    for i in range(len(centroid_list)):
        # Convert centroids to integers in order to index into image
        x = int(centroid_list[i][0])
        y = int(centroid_list[i][1])
        # Get string label from integer coordinates
        s = str(im_labeled[y][x])
        # Place region label text and store matplotlib Artist
        ax.text(
            x,
            y,
            s,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="white",
        )
    if im_overlay is not None:
        ax.imshow(im_overlay)
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close("all")
