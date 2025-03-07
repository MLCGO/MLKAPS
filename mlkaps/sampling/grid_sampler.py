"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

"""
Grid sampling using a samples count per dimensions
"""

import numpy as np
import pandas as pd
from deprecated import deprecated

from .static_sampler import StaticSampler


class GridSampler(StaticSampler):
    """
    A sampler that implements grid sampling using a number of samples per dimensions
    """

    def __init__(self, variable_types=None, variable_values=None, variable_mask=None):
        super().__init__(variable_types, variable_values, variable_mask)

    @staticmethod
    def _sample_linear_space(n_samples, variable_values, variable_type):
        dtype = "int" if variable_type == "int" else "float"

        return list(np.linspace(variable_values[0], variable_values[1], n_samples, dtype=dtype))

    def sample(self, n_samples: dict) -> pd.DataFrame:
        """
        Sample in a grid fashion.

        :param n_samples: A dict containing feature: n_samples for all variables to sample for
        :type n_samples: dict
        :raises ValueError: Raised when an unknown variable type is encountered
        :return: A DataFrame containing the new samples
        :rtype: pd.DataFrame
        """

        def generate_grid_samples(variable_values, variable_type, n_samples):
            if variable_type in ["Categorical", "Boolean"]:
                return variable_values
            elif variable_type in ["int", "float"]:
                return self._sample_linear_space(n_samples, variable_values, variable_type)
            else:
                raise ValueError(f"Unknown variable type: {variable_type}")

        variables = list(self.variables_types.keys())
        grids = [
            generate_grid_samples(self.variables_values[var], self.variables_types[var], n_samples[var]) for var in variables
        ]

        def recursive_sample(grids, depth=0, current_sample=[]):
            if depth == len(grids):
                yield current_sample
                return
            for value in grids[depth]:
                yield from recursive_sample(grids, depth + 1, current_sample + [value])

        samples = list(recursive_sample(grids))
        return pd.DataFrame(samples, columns=variables)
