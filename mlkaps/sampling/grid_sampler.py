"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

"""
Grid sampling using a samples count per dimensions
"""

import itertools

import numpy as np
import pandas as pd
from deprecated import deprecated

from .static_sampler import StaticSampler


# The current implementation uses itertools and generates the full space at once, which is highly costly for large samples count
@deprecated("Not maintained, may not work or be extremely slow.")
class GridSampler(StaticSampler):
    """
    A sampler that implements grid sampling using a number of samples per dimensions
    """

    def __init__(self, variable_types=None, variable_values=None, variable_mask=None):
        super().__init__(variable_types, variable_values, variable_mask)

    @staticmethod
    def _sample_linear_space(n_samples, variable_values, variable_type):
        dtype = "int" if variable_type == "int" else "float"

        return list(
            np.linspace(variable_values[0], variable_values[1], n_samples, dtype=dtype)
        )

    def sample(self, n_samples: dict) -> pd.DataFrame:
        """

        Sample in a grid fashion.

        :param n_samples: A dict containing feature: n_samples for all variables to sample for
        :type n_samples: dict
        :raises ValueError: Raised when an unknown variable type is encountered
        :return: A DataFrame containing the new samples
        :rtype: pd.DataFrame
        """

        to_iter = []

        for variable, variable_type in self.variables_types.items():
            # Aliases for readability
            variable_values = self.variables_values[variable]

            # Only two values for categorical/boolean features
            if variable_type in ["Categorical", "Boolean"]:
                to_iter.append(variable_values)
            # Integers/floats for continuous features
            elif variable_type in ["int", "float"]:
                feature_nsample = n_samples[variable]
                points = self._sample_linear_space(
                    feature_nsample, variable_values, variable_type
                )
                to_iter.append(points)
            else:
                raise ValueError("GridSampler - Unknown feature type")

        # Create a dataframe containing all the samples
        # that needs to be executed
        # Fixme: this is the biggest bottleneck in the grid sampler, where we must
        # create a dataframe from the product of all dimensions of the grid
        # This is really slow with a large number of dimensions/samples per dimension
        res = pd.DataFrame(list(itertools.product(*to_iter)))
        res.columns = self.variables_values.keys()

        return res
