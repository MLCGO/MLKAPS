"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pandas as pd

_encoding = {
    "Categorical": "category",
    "int": "int64",
    "float": "float64",
    "Boolean": "bool",
}


def encode_dataframe(dtypes: dict, dataframe: pd.DataFrame):
    """
    This function iterate on the columns of a dataframe,
    and assigns the correct type to each column.

    Parameters
    ----------
    dtypes: dict
        A dictionary containing the type of each column.
        If a column is not in the dictionary, it is left unchanged.

    dataframe: pd.DataFrame
        The dataframe to encode.

    Returns
    -------
    pd.DataFrame
        An encoded dataframe
    """
    mapping = {}
    for feature in dataframe.columns:
        if feature not in dtypes:
            continue
        ftype = dtypes[feature]
        mapping[feature] = _encoding[ftype]
    encoded_dataframe = dataframe.astype(mapping)
    return encoded_dataframe
