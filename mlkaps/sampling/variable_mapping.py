"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Utilities for mapping variables of different types (int, float, categorical, ...) to numerical values and back
"""

import numpy as np
import pandas as pd


def _check_variables_type(variables):
    for variable, var_type in variables.items():
        if var_type not in ["Categorical", "Boolean", "float", "int"]:
            raise ValueError(
                f"Unrecognized variable type for variable {variable}: {var_type}. Type must be one of ['Categorical', 'Boolean', 'float', 'int']"
            )


def map_variables_to_numeric(data: pd.DataFrame, variables_types: dict, variables_values: dict) -> pd.DataFrame:
    """
    Map a dataframe containing integers/categorical/boolean values to the corresponding numeric values
    as defined by the variables_values/types dictionaries

    The mapping can be reversed using the map_float_to_variables function.


    :param data: The dataframe to map to float values
    :type data: pandas.DataFrame
    :param variables_types:
        A dictionary associating the name of each variable to its type. The type can be either
        ["Categorical", "Boolean", "int", "float"]
    :type variables_types: dict
    :param variables_values:
        A dictionary associating the name of each variable to its possible values. The possible
        values must be a list of values for categorical variables, or a tuple (min, max) for
        numerical variables.
    :type variables_values: dict

    :return: The dataframe with the variables mapped to float values
    :rtype: pandas.DataFrame
    :raise ValueError: If one of the variables types is unrecognized
    """

    # Ensure the variables have a valid type
    _check_variables_type(variables_types)

    # Copy the data to avoid side-effects
    data = data.copy()
    for variable, var_type in variables_types.items():
        if var_type in ["Categorical", "Boolean"]:
            # We map every category to one integer using a map
            feature_map = {k: j for j, k in enumerate(variables_values[variable])}
            data[variable] = np.vectorize(feature_map.get)(data[variable])

        elif var_type == "int":
            # Round and use the correct type
            data[variable] = data[variable].astype(float)

    return data


def map_float_to_variables(data: pd.DataFrame, variables_types: dict, variables_values: dict):
    """
    Map a dataframe containing float values to the corresponding integer/categorical/boolean values
    as defined by the variables_values/types dictionaries

    Reverse operation of map_variables_to_float


    :param data: The dataframe to map to float values
    :type data: pandas.DataFrame
    :param variables_types:
        A dictionary associating the name of each variable to its type. The type can be either
        ["Categorical", "Boolean", "int", "float"]
    :type variables_types: dict
    :param variables_values:
        A dictionary associating the name of each variable to its possible values. The possible
        values must be a list of values for categorical variables, or a tuple (min, max) for
        numerical variables.
    :type variables_values: dict

    :return: The dataframe with the variables mapped to integer/categorical/boolean values
    :rtype: pandas.DataFrame
    :raise ValueError: If one of the variables types is unrecognized
    """

    # Ensure the variables have a valid type
    _check_variables_type(variables_types)

    # Copy the data to avoid side-effects
    data = data.copy()
    for variable, var_type in variables_types.items():
        if var_type in ["Categorical", "Boolean"]:
            # Round and use the correct type
            # Note that rounding is important, otherwise values might be mapped to the wrong
            # category
            data[variable] = np.round(data[variable]).astype(int)

            # We map the integer values to the corresponding
            # Categorical/Boolean using a map
            feature_map = {j: k for j, k in enumerate(variables_values[variable])}

            # Then converts all those integers to their corresponding
            # values
            data[variable] = np.vectorize(feature_map.get)(data[variable])

        elif var_type == "int":
            # Round and use the correct type
            data[variable] = np.round(data[variable]).astype(int)

    return data
