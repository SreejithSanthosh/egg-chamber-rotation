"""Utility functions for selecting paths."""

import os
import re


def select_files(input_dir, file_labels):
    """
    Find a set of files from an input directory.

    Written to make it easier to grab processed image files.

    Parameters
    ----------
    input_dir : str
        Path to a directory with input files
    file_labels : str or list of str
        String to find after basename in file, or list of such strings.
        Each one must complete the file name when appended to a basename.

    Returns
    -------
    out_dict_ls : list of dicts
        Checks all dataset in input_dir. For each one, if a file is present
        for each element of file_labels, then a dict is added to out_dict_ls
        for that dataset. This dict always includes at least two keys:
            "basename" : str name of the basefile for the dataset
            "basefile" : str path to the basefile for the dataset
        It also includes a key for each element of file_labels. For each one
        the value is a str of the path to the matching file
    """
    if isinstance(file_labels, str):
        file_labels = [file_labels]

    out_dict_ls, basenames = [], []

    # Make a list of all basenames in the directory
    with os.scandir(input_dir) as input_dir_path_ls:
        for item in input_dir_path_ls:
            basename = get_basename(item)
            if basename and not item.is_dir():
                basenames.append(basename)
    basenames_unique = list(set(basenames))

    # If the basefile can be found for a basename, add a dict to out_dict_ls
    input_dir_ls = os.listdir(input_dir)
    for basename in basenames_unique:
        dict_tmp = {"basename": basename}
        for lab in file_labels:
            file_to_check = f"{basename}{lab}"
            if file_to_check in input_dir_ls:
                dict_tmp[lab] = os.path.join(input_dir, file_to_check)
        if f"{basename}.czi" in input_dir_ls:
            dict_tmp["basefile"] = os.path.join(input_dir, f"{basename}.czi")
        elif f"{basename}.tif" in input_dir_ls:
            dict_tmp["basefile"] = os.path.join(input_dir, f"{basename}.tif")
        if len(dict_tmp.keys()) == len(file_labels) + 2:
            out_dict_ls.append(dict_tmp)

    return out_dict_ls


def get_basename(path):
    """
    Get the basename from a Path object.

    Here basename is defined as this format:
        [alphanumeric string]_[digits]

    It is expected that this will be at the beginning of a file name.

    Parameters
    ----------
    path : Path object

    Returns
    -------
    None if no match is found, or basename as a str if there is a match.
    """
    m = re.search("^\w+_\d+(?=[\._])", path.name)
    if m is None:
        return None
    else:
        return m[0]
