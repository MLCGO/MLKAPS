"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

"""
Contain the definition for static samplers that only need an array containing the bounds of each variable.
"""

import numpy as np
import pandas as pd
from smt.sampling_methods import Random, LHS

from .sampler import SamplerError
from .static_sampler import StaticSampler
from .variable_mapping import map_float_to_variables


def convert_variables_bounds_to_numeric(
    variables_types, variables_values, include_high_bound=False
):
    """
    Convert a dictionary of variables bounds to a dictionary of numeric bounds:
    Categorical variables are converted to [0, n_categories - 1] if include_high_bound is False,
    or [0, n_categories] if include_high_bound is True.

    For integer and float variables, the bounds are set to [min, max] where min and max are the
    lowest and highest possible values for the variable.


    :param variables_types: A dictionary associating the name of each variable to its type. The type can be either
        "Categorical", "Boolean", "int" or "float".
    :type variables_types: dict
    :param variables_values: A dictionary associating the name of each variable to its possible values. The possible
        values must be a list of values for categorical variables, or a tuple (min, max) for
        numerical variables.
    :type variables_values: dict
    :param include_high_bound: Whether the upper bounds for categorical variables should be inclusive or not.
        i.e, if True, the upper bound is set to n_categories, else it is set to n_categories - 1.
    :type include_high_bound: bool

    :return: A dictionary associating the name of each variable to its bounds.
    :rtype: dict
    """

    # Generate a list of bounds for each parameter
    bounds = {}
    for variable, var_type in variables_types.items():
        # For boolean/Categorical, just take the number of possible values
        if var_type in ["Categorical", "Boolean"]:
            if not include_high_bound:
                bounds[variable] = [0, len(variables_values[variable]) - 1]
            else:
                bounds[variable] = [0, len(variables_values[variable])]
        else:
            bounds[variable] = variables_values[variable]
    return bounds


class GenericBoundedSampler(StaticSampler):
    """
    A sampler that works with any sampling techniques based on an array of bounds.
    Conceived with smt.sampling_methods in mind (LHS, Random, etc.)
    """

    def __init__(
        self,
        generic_sampler_type,
        variable_types=None,
        variable_values=None,
        variable_mask=None,
    ):
        """
        Build a new generic sampler using the generic_sampler_type sampling method


        :param generic_sampler_type: A sampling object that uses smt API;
            - Must have a constructor with a xlimits argument, defining a list of list,
            corresponding to the bounds of each variable
            - Must have a __call__ method, returning 2d list of samples
        :param variable_types: A dictionary associating the name of each variable to its type. The type can be either
            "Categorical", "Boolean", "int" or "float".
            Can be None, in which case the bounds are not generated, and must be set later using
            set_variables(...)
        :type variables_types: dict
        :param variable_values: A dictionary associating the name of each variable to its possible values. The possible
            values must be a list of values for categorical variables, or a tuple (min, max) for
            numerical variables.
            Can be None, in which case the bounds are not generated, and must be set later using
            set_variables(...)
        :type variable_values: dict
        :param variable_mask:
            A list of names of variables to sample. If None, all variables are sampled. Else,
            only the variables in the list are sampled.
            Can be useful when the variables to samples are a subset of the variables in
            variable_types
        :type variable_mask: list
        """

        self.sampler_type = generic_sampler_type
        # Calling the parent constructor will call set_variables()
        # We must define the bounds to be empty before
        self.bounds = None
        super().__init__(variable_types, variable_values, variable_mask)

    def _generate_bounds(self):
        """
        Generate the bounds of the sampling process

        :return: A dictionnary containing the bounds for each variables
        :rtype: dict(str, list)
        """

        if self.variables_types is None or self.variables_values is None:
            return None
        return convert_variables_bounds_to_numeric(
            self.variables_types, self.variables_values
        )

    def set_variables(self, variables_types, variables_values, mask=None):
        """
        Set the variables used in the sampling process.

        :param variables_types: Contains the types for each variable, must be one of ["int", "float", "Boolean", "Categorical"]
        :type variables_types: dict
        :param variables_values: Contain the possible values for each variable:
            For continuous types (int, float), must be a range [min, max]
            For Categorical/Boolean types, must be a list of possible values
        :type vairables_values: dict
        :param mask: An iterable containing a list of variables to consider during the sampling process
            Variables not contained in the mask will be ignored
        """

        super().set_variables(variables_types, variables_values, mask)
        self.bounds = self._generate_bounds()

    def _generate_samples_from_bounds(self, n_samples: int):
        """
        Execute the sampler on the bounded variable space

        :raise SamplerError: raise an exception if the variables were not set before usage, or if the sampler failed

        :param n_samples: The number of samples to take
        :type n_samples: int

        :return: A list of samples
        :rtype: pandas.DataFrame
        """

        self._raise_if_variables_not_set()

        # Dict are not guaranteed to be ordered
        # This may cause an issue where the samples do not match the order of the variables
        # As a safety measure, we sort the bounds dict by keys to guarantee the
        # order
        ordered_key = sorted(self.bounds.keys())
        fixed_order_bounds = np.array([self.bounds[i] for i in ordered_key])

        try:
            sampler = self.sampler_type(xlimits=fixed_order_bounds)
            # The sampler returns a dict of array containing every value for each
            # parameter
            random_samples = sampler(n_samples)
            # Create a new dataframe to ensure ordering
            ordered_random_samples = pd.DataFrame(random_samples, columns=ordered_key)
        except Exception as exc:
            raise SamplerError from exc

        return ordered_random_samples

    def sample(self, n_samples: int) -> pd.DataFrame | None:

        if n_samples == 0:
            return None

        if n_samples < 0:
            raise SamplerError(f"Cannot sample negative samples count ({n_samples}) !")

        random_samples = self._generate_samples_from_bounds(n_samples)

        # Map the generated samples with numeric features back to the original variables types
        translated_columns = map_float_to_variables(
            random_samples, self.variables_types, self.variables_values
        )

        columns = sorted(self.bounds.keys())

        # Now that we translated each column, we can create the final dataframe
        translated_samples = pd.DataFrame(translated_columns, columns=columns)

        return translated_samples


class LhsSampler(GenericBoundedSampler):
    """
    Sampler based on Latin Hypercube Sampling.
    """

    def __init__(self, variable_types=None, variable_values=None, variable_mask=None):
        super().__init__(LHS, variable_types, variable_values, variable_mask)


class RandomSampler(GenericBoundedSampler):
    """
    Sampler based on random uniform sampling
    """

    def __init__(self, variable_types=None, variable_values=None, variable_mask=None):
        super().__init__(Random, variable_types, variable_values, variable_mask)
