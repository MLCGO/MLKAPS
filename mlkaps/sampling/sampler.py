"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Define the base class for all sampler
"""


def _mask_variables(variables: dict, mask: list) -> dict:
    """
    Helper function to filter out variables that are not in the mask.


    :param variables: The variables to filter
    :type variables: dict
    :param mask: The list of variables to keep
    :type mask: list


    :return: Masked dictionary
    :dict: dict
    """
    if mask is None:
        return variables
    return {key: value for key, value in variables.items() if key in mask}


class SamplerError(Exception):
    """
    Generic exception to raise when a sampler fails
    """


class Sampler:
    """
    Base class for all samplers.
    """

    def __init__(self, variables_types=None, variables_values=None, variables_mask=None):
        """
        Initializes the sampler.

        :param variables_types: A dictionary associating the name of each variable to its type. The type can be either
            ["Categorical", "Boolean", "int", "float"]. If None, then the variables types must be
            set using the set_variables method before sampling.
        :type variables_types:  dict
        :param variables_values:
            A dictionary associating the name of each variable to its possible values.
            Continuous variables (int, float) must be a range [min, max]
            Categorical/Boolean variables must be a list of possible values
        :type variables_values:  dict
        """
        self.variables_values = None
        self.variables_types = None

        # Set the variables using a setter to ensure that overriding classes can perform
        # additional checks if needed
        self.set_variables(variables_types, variables_values, variables_mask)

    def _raise_if_variables_not_set(self):
        """
        :raise SamplerError: Raise an exception if the variables are not set,
        or if the variables are empty (after masking for example).
        """

        if self.variables_values is None or self.variables_types is None:
            raise SamplerError("The sampler variables (values and/or types) were not set!")
        if len(self.variables_values) == 0 or len(self.variables_types) == 0:
            raise SamplerError("The passed variables were empty, or all variables were masked out!")

    def set_variables(self, variables_types: dict, variables_values: dict, mask: list = None):
        """
        Sets the variables to be sampled.

        :param variables_types:
            A dictionary associating the name of each variable to its type. The type can be either
            ["Categorical", "Boolean", "int", "float"].
            If none, then the variables are cleared and must be set again before sampling.
        :type variables_types: dict
        :param variables_values:
            A dictionary associating the name of each variable to its possible values. The possible
            values must be a list of values for categorical variables, or a tuple (min, max) for
            numerical variables.
            If none, then the variables are cleared and must be set again before sampling.
        :type variables_values: dict
        :param mask:
            A list of variables to keep. If None, all variables are kept. This can be useful when
            the variables to be samples are a subset of the variables defined in the sampler.
        :type mask: list
        """

        # Filter out masked variables
        variables_values = _mask_variables(variables_values, mask)
        self.variables_values = variables_values

        variables_types = _mask_variables(variables_types, mask)
        self.variables_types = variables_types

        if variables_values is None or variables_types is None:
            return

        # Ensure that both dict contain the same keys (variables)
        assert sorted(self.variables_types.keys()) == sorted(self.variables_values.keys())
