"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

"""
Definition for the base class of all adaptive samplers
"""

from collections.abc import Callable
import pandas as pd
from ..sampler import Sampler


class AdaptiveSampler(Sampler):
    """
    Base class for all adaptive samplers
    This API is designed so that stopping criterion are handled separately from the sampling logic
    This class is stateful, and should be initialized with init() before sampling
    This is made so the sampler can track custom metrics and dump them to disk if needed
    """

    def __init__(
        self,
        variables_types: dict = None,
        variables_values: dict = None,
        mask: list = None,
    ):
        """
        Initialize the sampler

        :param variables_types:
            The types of the variables to be sampled, as a dict of {variable_name: variable_type}
            The type can be either ["int", "float", "Categorical", "Boolean"]
            If none is provided, the variables must be set later with set_variables(...)
        :type variables_types: dict
        :param variables_values:
            The values of the variables to be sampled, as a dict of {variable_name: variable_values}
            The values must be a tuple of (min, max) for numerical variables, or a list of
            possible values for categorical variables
            If none is provided, the variables must be set later with set_variables(...)
        :type variables_values: dict
        :param mask:
            A list of variables to filter out from the variables_types and variables_values
            If none is provided, all variables are kept
        :type mask: list
        """

        super().__init__(variables_types, variables_values, mask)

    def reset(self):
        """
        Reset and initialize the sampler before a new sampling process
        """

        raise NotImplementedError()

    def sample(
        self,
        n_samples: int,
        data: pd.DataFrame | None,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Sample n_samples from execution_func, and append them to data

        :param n_samples:
            Number of samples to be drawn
        :type n_samples: int
        :param data:
            The already sampled data, or None
            If none, the sampler should bootstrap the data with its own strategy
        :type data: pandas.DataFrame
        :param execution_func:
            The function to be executed on the sampled data
            This function MUST return the original dataframe, with the resulting columns appended
        :type execution_function: Callable[[pandas.DataFrame], pandas.DataFrame]

        :return: A dataframe containing the sampled points with their results
        :rtype: pandas.DataFrame
        """

        raise NotImplementedError()

    def dump(self, output_directory):
        """
        Dump the sampler to the output directory, for debugging or analysis of the sampling process


        :param output_directory:
            The directory to dump the sampler to
        """

        raise NotImplementedError()
