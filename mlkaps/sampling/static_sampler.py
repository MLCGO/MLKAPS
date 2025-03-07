"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Define StaticSampler, the base class for all static samplers (Samplers where the final number of samples is known when called)
"""

import pandas as pd

from .sampler import Sampler


class StaticSampler(Sampler):
    """
    Base class for all samplers where the number of samples is fixed (static) when called, as opposed to
    adaptive samplers where the number of samples is not known in advance and also depends on
    previously sampled values.
    """

    def __call__(self, n_samples) -> pd.DataFrame | None:
        """
        Run the sampler

        :return: The list of samples
        :rtype: pandas.DataFrame
        :raise SamplerError: If the sampling process failed
        """

        return self.sample(n_samples)

    def sample(self, n_samples: int) -> pd.DataFrame | None:
        """
        Samples n_samples from the defined variables, returns a dataframe with the sampled values.


        :param n_samples: The number of samples to draw
        :type n_samples: int

        :return: A dataframe containing the sampled values, or None if 0 samples were drawn.
        :rtype: pd.DataFrame
        :raise SamplerError: If the sampling process failed
        """

        raise NotImplementedError()
