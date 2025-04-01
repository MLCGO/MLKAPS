"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pandas as pd
from typing import Callable


class FailedRunResolver:
    """
    Base class for all strategies to deal with failed runs during sample collection
    """

    # Keep track of all known resolvers
    known_resolvers = {}

    def __init_subclass__(cls, resolver_name: str | None = None) -> None:
        """
        Register a new resolver with a given name

        :param resolver_name: The name of the resolver, if not specified, will default to the class name
        :type resolver_name: str, optional
        :raises KernelSamplingError: Raised if a name conflict is detected
        """

        if resolver_name is None:
            resolver_name = cls.__name__

        if resolver_name in cls.known_resolvers:
            raise ValueError(
                f"Conflicting resolver name class <{cls.__name__}> declared the name '{resolver_name}' \
                which is already in use by <{cls.known_resolvers[resolver_name].__name__}>"
            )

        cls.known_resolvers[resolver_name] = cls

    @staticmethod
    def from_name(name: str, *args, **kwargs) -> Callable:
        resolver_type = FailedRunResolver.known_resolvers.get(name, None)
        if resolver_type is None:
            raise ValueError(f"Unknown resolver '{name}'")

        return resolver_type(*args, **kwargs)


class DiscardResolver(FailedRunResolver, resolver_name="discard"):
    """
    Drop all samples with missing values (NaNs) from the dataset
    """

    def __call__(self, samples: pd.DataFrame, subset=None) -> pd.DataFrame:
        """
        Apply the resolver to the given Dataframe, optionally specify a subsets of columns to replace

        :param samples: The samples to resolve
        :type samples: pd.DataFrame
        :param subset: A list of columns to consider, defaults to all columns
        :type subset: list[str], optional
        :return: The filtered dataset
        :rtype: pd.DataFrame
        """

        if subset is not None and any(col not in samples.columns for col in subset):
            raise ValueError(f"Invalid subset of columns. Received: {subset}. Available: {samples.columns}")

        return samples.dropna(subset=subset)


class ConstantResolver(FailedRunResolver, resolver_name="constant"):
    """
    Replace all samples with missing values (NaNs) with a constant value
    """

    def __init__(self, value: float | int):
        """
        Build a new constant resolver using the specified value

        :param value: The value to use during resolution
        :type value: float | int
        """
        self.constant = value

    def __call__(self, samples: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
        """
        Apply the resolver to the given Dataframe, optionally specify a subsets of columns to replace

        :param samples: The samples to resolve
        :type samples: pd.DataFrame
        :param subset: A list of columns to consider, defaults to all columns
        :type subset: list[str], optional
        :return: The filtered dataset
        :rtype: pd.DataFrame
        """

        if subset is not None and any(col not in samples.columns for col in subset):
            raise ValueError(f"Invalid subset of columns. Received: {subset}. Available: {samples.columns}")

        # Replace all nans or only a subset of columns
        subset = subset or samples.columns
        replacement = {col: self.constant for col in subset}
        return samples.fillna(replacement)
