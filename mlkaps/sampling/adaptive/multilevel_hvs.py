"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from collections.abc import Callable

import pandas as pd

from .adaptive_sampler import AdaptiveSampler
from .hvs import HVSampler


class MultilevelHVS(AdaptiveSampler):
    """
    Multilevel version of the Hierarchical Variance Sampling (HVS) Algorithm
    This sampler recursively partitions the data based on multiple levels of features
    This is equivalent to decision tree partitioning with constraints on the splitting criterion
    ordering.

    When the final level is reached, the HVS algorithm is used to sample the data based on
    the final partitions

    """

    def __init__(
        self,
        features_levels: list[list] = None,
        variables_types=None,
        variables_values=None,
    ):
        self.hvs = HVSampler(variables_types, variables_values, "cov")

        super().__init__(variables_types, variables_values)
        self.features_levels = features_levels
        self.partitions = []

    def reset(self):
        return

    def dump(self, output_directory):
        return

    def set_per_level_features(self, leveled_features: list[list]):
        self.features_levels = leveled_features

    def set_variables(self, variables_types, variables_values, mask=None):
        super().set_variables(variables_types, variables_values, mask)
        self.hvs.set_variables(variables_types, variables_values, mask)

    def sample(
        self,
        n_samples: int,
        data: pd.DataFrame | None,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:

        if self.features_levels is None:
            raise Exception("Features levels not set")

        # Handle empty data (bootstrap)
        if data is None or len(data) == 0:
            return self.hvs.sample(n_samples, data, execution_func)
        # The objectives are the columns that are not labelled as features
        objectives = [k for k in data.columns if k not in self.variables_values.keys()]
        n_samples_per_objective = max(1, n_samples // len(objectives))

        # We sample separately for each objective
        new_samples = None
        self.partitions = []
        for objective in objectives:
            # Partition the data for current objective
            objective_samples = self._partition(objective, data, self.features_levels, n_samples_per_objective)
            new_samples = pd.concat([new_samples, objective_samples], axis=0, ignore_index=True)

        # Run the execution function on the new samples
        labelled_samples = execution_func(new_samples)
        data = pd.concat([data, labelled_samples], axis=0, ignore_index=True)

        return data

    def _partition(
        self,
        objective,
        data: pd.DataFrame,
        leveled_features: list[list],
        n_samples: int,
        axes=None,
    ) -> pd.DataFrame:

        # Run HVS based on the features in the current level
        features = {k: v for k, v in self.variables_values.items() if k in leveled_features[0]}
        next_features = leveled_features[1:]

        # Even if we're cutting on some different axis, we need to propagate all the axis
        # limitations coming from the previous levels
        # For example, if the current partition has A = [0, 2.5]
        # But we're cutting on B, we still need to respect the range for A
        if axes is None:
            axes = self.variables_values.copy()

        new_samples = None
        # We reached the last level, partition based on current features, and samples using HVS
        if len(next_features) == 0:
            new_samples = self._partition_final(axes, data, features, n_samples, objective)
        else:
            # We are not in the last level, partition based on current features, and apply the
            # next level on each partition
            new_samples = self._partition_intermediate(axes, data, features, n_samples, next_features, objective)

        return new_samples

    def _partition_final(self, axes, data, features, n_samples, objective):

        # First, partition using current features
        partitions, _ = self.hvs.partition(data, objective, n_samples, split_on=features, min_samples_per_leaf=15)

        # The current partitions are not necessarily covering all the axes limitations
        # We merge the axes limitations of previous levels with the current partitions
        self._merge_axes(axes, partitions)

        self.partitions.extend([p[1] for p in partitions])
        return self.hvs.sample_partitions(partitions)

    def _merge_axes(self, axes, partitions):
        # Merge the axes defined in the partition with the axes defined in the previous levels
        for partition in partitions:
            partition = partition[1]
            for axis in axes:
                if axis not in partition.axes.keys():
                    partition.axes[axis] = axes[axis]

    def _partition_intermediate(self, axes, data, features, n_samples, next_features, objective):
        # First, partition using current features
        # The minimum number of samples per leaf must be high enough so that we can split on the
        # next level

        partitions, _ = self.hvs.partition(data, objective, n_samples, split_on=features, min_samples_per_leaf=90)

        # The current partitions are not necessarily covering all the axes limitations
        # We merge the axes limitations of previous levels with the current partitions
        self._merge_axes(axes, partitions)

        res = None
        # Then, apply the next level on all partitions
        for p in partitions:

            n_samples = p[0]
            partition = p[1]

            # No need to continue if no samples is allocated to the partition, or if the
            # current partition is empty
            if n_samples == 0 or partition.samples is None or len(partition.samples) == 0:
                continue

            results = self._partition(
                objective,
                partition.samples,
                next_features,
                n_samples,
                axes=partition.axes,
            )
            res = pd.concat([res, results], axis=0, ignore_index=True)
        return res
