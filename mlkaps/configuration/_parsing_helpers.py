"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import os.path


def set_if_defined(key, json, output_dict, default_value=None, subkey=None):
    """
    Search if key exist in the input dict, and set it accordingly in the output dict
    if found. Else, set it to the default value if provided.

    Parameters
    ----------
    key : str
        The key to look for in the json

    json:
        The dictionary to look for the key in

    output_dict:
        The dictionary to set the key in if it is found in the input dictionary

    default_value:
        The default value to use if the key is not found in the input dictionary

    subkey:
        If provided, set [key][subkey] = value instead of [key] = value

    """
    v = None
    if key in json:
        v = json[key]
    elif default_value is not None:
        v = default_value

    if v is None:
        return

    # If a subkey is provided, then set the subkey of the main key to the
    # value (if any) map { "key": { "subkey": "value" }
    if subkey is not None:
        output_dict[key][subkey] = v
    else:
        output_dict[key] = v


def check_file_is_executable(path):
    """
    Check if a file exists and is executable

    Parameters
    ----------
    path : str
        The path of the file to check for

    Returns
    -------
    bool
        True if the file exists and is executable, False otherwise

    """
    return os.path.isfile(path) and os.access(path, os.X_OK)


def assert_file_is_executable(path):
    """
    Assert that a file exists and is executable

    Parameters
    ----------
    path : str

    Raises
    ------
    Exception
        If the file doesn't exist or is not executable
    """
    if not check_file_is_executable(path):
        raise Exception(
            "Expected executable file, file either doesn't exist, or is not executable:"
            "{}".format(path)
        )
