"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Definition of the Hiearchical Variance Sampling (HVS) adaptive sampling method
based on DOI:10.1007/978-3-642-32820-6_11
"""

from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from numpy import ceil, sqrt
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
import textwrap
from smt.sampling_methods import Random

from .adaptive_sampler import AdaptiveSampler
from ..sampler import SamplerError
from mlkaps.sampling.variable_mapping import (
    map_variables_to_numeric,
    map_float_to_variables,
)
from ..generic_bounded_sampler import LhsSampler
from ..generic_bounded_sampler import convert_variables_bounds_to_numeric


def _error_variance(confidence: float, input: pd.DataFrame) -> float:
    """
    Computes an upper bound of the variance of the input dataframe

    :param confidence_interval: The desired confidence (alpha) for the variance.
        For a confidence interval of 95%, this value should be 0.05
    :type confidence_interval: float
    :param input: The data on which to compute the variance upper bound
    :type input: A DataFrame
    :return: An upper bound of the input's variance
    :rtype: float
    """
    # Compute the variance with correction
    est_variance = input.var()

    # Note: stats.chi.ppf(0.05)**2 computes chi^2(1-0.05) !
    card = len(input)
    variance_corr = (card - 1) / stats.chi2.ppf(confidence / 2, card - 1)

    return est_variance * variance_corr


def _error_cov(confidence: float, input: pd.DataFrame) -> float:
    """
    Computes an upper bound of the covariance of the input dataframe

    :param confidence_interval: The desired confidence (alpha) for the covariance.
        For a confidence interval of 95%, this value should be 0.05
    :type confidence_interval: float
    :param input: The data on which to compute the covariance upper bound
    :type input: A DataFrame
    :return: An upper bound of the input's covariance
    :rtype: float
    """

    # Coefficient of variation error
    card = len(input)
    # Compute the variances for all columns
    est_variance = input.var()
    # Consider the column with the highest variance
    mean = input.mean()

    # Compute Variance upper bound
    variance_corr = (card - 1) / stats.chi2.ppf(confidence / 2, card - 1)
    var_ub = est_variance * variance_corr
    if var_ub <= 1e-6 or abs(mean) < 1e-6:
        return 0

    # Compute covariance upper bound
    cov_ub = sqrt(var_ub) / abs(mean)

    # Return the upper bound of the covariance
    return cov_ub**2


_known_error_functions = {"cov": _error_cov, "variance": _error_variance}


def _get_error_function(name: str) -> Callable[[pd.DataFrame], float]:
    """
    Fetch an error function by name

    :param confidence: The confidence to use in the error function
    :type confidence: float
    :param name: The name of the error function
    :type name: str
    :raises SamplerError: Raised if the name doesn't correspond to a known error function
    :return: The requested error function
    :rtype: Callable[[pd.DataFrame], float]
    """

    # This is a constant for now
    confidence = 0.05
    if name not in _known_error_functions:
        raise SamplerError(f"Unknown error function for HVS: '{name}'")
    return lambda x: _known_error_functions[name](confidence, x)


class HVSPartition:
    """
    Helper class to store partitions necessary for the HVS algorithm

    Provides utilities to compute a partition size and error
    """

    def __init__(
        self,
        bounds: dict,
        samples: pd.DataFrame,
        node_id: int,
        error_function: Callable[[pd.DataFrame], pd.DataFrame],
        objective: str,
        size_metric: Callable,
    ):
        """
        Create a new HVS Partition

        :param bounds:
            A dictionnary containing the bounds of the partitions for each feature,
            stored as feature: [low_bound, high_bound]
        :type bounds: dict
        :param samples:
            A DataFrame containing the samples in this partition
        :type samples: pd.DataFrame
        :param node_id:
            The id of the node in the DecisionTree corresponding to this leaf
        :type node_id: int
        :param error_function:
            The function to use to compute the error in this partition
        :type error_function: Callable[[pd.DataFrame], pd.DataFrame]
        :param objective:
            The name of the features in the samples to consider as
            the objective to compute the error
        :type objective: str
        :param size_metric:
            The method used to compute the size of the partition
        :type size_metric: Callable[[HVSPartition], float], optional
        """

        self.axes = bounds
        self.samples = samples
        self.node_id = node_id

        self.size = size_metric(self)
        self.error = error_function(self.samples[objective])
        self.score = self.size * self.error

    def __str__(self):
        msg = f"Bounds: {self.axes}\nSamples:\n{self.samples} \
            \n\nSize: {self.size}\nError: {self.error}\nScore: {self.score}"
        msg = textwrap.indent(msg, "\t")
        return f"HVS Partition (Node #{self.node_id}):\n{msg}"


def _std_partition_size(partition: HVSPartition) -> float:
    """
    Standard metric to compute a partition size, multiply each axis length

    :param partition: The partition to compute the size for
    :type partition: HVSPartition
    :return: The size of the partition
    :rtype: float
    """
    size = 1
    for axis in partition.axes.values():
        size *= axis[1] - axis[0]

    return size


def _partition_density_size(partition: HVSPartition) -> float:
    """
    Density metric for a partition size, divide the standard metric by the number of samples

    :param partition: The partition to compute the size for
    :type partition: HVSPartition
    :return: The size of the partition
    :rtype: float
    """
    size = _std_partition_size(partition) / len(partition.samples)

    return size


_known_size_metrics = {"size": _std_partition_size, "density": _partition_density_size}


def _get_size_metric(name):
    if name not in _known_size_metrics:
        raise SamplerError(f"Unknown size metric '{name}'")

    return _known_size_metrics[name]


class HVSPartitionner:
    """
    Helper class to partition a set of samples using a decision tree,
    according to the HVS strategy
    """

    def __init__(
        self,
        features: dict,
        objective: str,
        tree_params: dict,
        error_metric: str | Callable[[pd.DataFrame], float] = "variance",
        size_metric: str | Callable[[HVSPartition], float] = "size",
    ):
        """
        Construct a new HVSPartitionner

        :param features: A dictionnary containing all the features to use to partition the samples
        :type features: dict
        :param objective: The name of the features in the samples to consider as the objective
        :type objective: str
        :param tree_params: The parameters to pass to the DecisionTreeRegressor
        :type tree_params: dict
        :param error_metric:
            The error metric to use to compute the error of each partition, defaults to "variance"
            If a string is passed, the partitionner will try to fetch the corresponding metric.
            Callables are used directly.
        :type error_metric: str | Callable[[pd.DataFrame], float], optional
        :param size_metric:
            The size metric to use to compute the size of each partition, defaults to "size"
            If a string is passed, the partitionner will try to fetch the corresponding metric.
            Callables are used directly.
        :type size_metric: str | Callable[[HVSPartition], float], optional
        """

        self.ordered_features = sorted(list(features.keys()))
        self.features = {k: features[k] for k in self.ordered_features}
        self.objective = objective
        self.tree_params = tree_params

        if isinstance(error_metric, str):
            error_metric = _get_error_function(error_metric)
        self.error_metric = error_metric

        if isinstance(size_metric, str):
            size_metric = _get_size_metric(size_metric)
        self.size_metric = size_metric

        self.tree: DecisionTreeRegressor = None

    def partition(self, samples: pd.DataFrame) -> list[HVSPartition]:
        """
        Run the partitionner on the given samples

        :param samples: The samples to partition
        :type samples: pd.DataFrame
        :return: A list of the built partitions
        :rtype: list[HVSPartition]
        """
        self.tree = self._build_tree(samples)
        partitions = self._build_partitions(self.tree, samples)

        return partitions

    def _build_tree(self, samples: pd.DataFrame):
        """
        Build the decision tree and fit it on the samples

        :param samples: The samples to fit the tree on
        :type samples: pd.DataFrame
        :return: The fitted decision tree
        :rtype: DecisionTreeRegressor
        """
        model = DecisionTreeRegressor(**self.tree_params)
        model.fit(samples[self.ordered_features], samples[self.objective])

        return model

    def _make_node_children(
        self,
        tree_features: dict,
        thresholds: dict,
        children_left: dict,
        children_right: dict,
        parent_node: tuple[int, dict],
    ) -> list:
        """
        Fetch the two children of a node in the decision tree and computes their bounds

        :param tree_features:
            A dictionnary containing the feature used at each node
        :type tree_features: dict
        :param thresholds:
            The threshold used at each node of the tree
        :type thresholds: dict
        :param children_left:
            The left children of each node
        :type children_left: dict
        :param children_right:
            The right children of each node
        :type children_right: dict
        :param node:
            The parent node id and its bounds
        :type node: tuple[int, dict]
        :return: A list containing a tuple (node_id, bounds) for the two children of the parent node
        :rtype: list
        """
        node_id = parent_node[0]
        bounds = parent_node[1]

        # Compute the new constraints for the children
        feature = self.ordered_features[tree_features[node_id]]
        threshold = thresholds[node_id]

        # Build the children and insert them at the beginning of the stack
        left_axes = bounds.copy()
        left_axes[feature] = [bounds[feature][0], threshold]
        left_child = (children_left[node_id], left_axes)

        right_axes = bounds.copy()
        right_axes[feature] = [threshold, bounds[feature][1]]
        right_child = (children_right[node_id], right_axes)

        return [left_child, right_child]

    def _parse_tree(self, tree: DecisionTreeRegressor):
        """
        Parse the tree and yield a tuple (node_id, bounds) when reaching a leaf

        :param tree: The tree to parse
        :type tree: DecisionTreeRegressor
        :yield:
            A leaf node id in the tree, and a dictionnary containing the bounds for each feature
        :rtype: tuple[int, dict]
        """

        # The features are addressed by their index in the features array
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        queue = [(0, self.features)]
        # Parse the tree depth first
        while len(queue) > 0:
            node = queue.pop()
            node_id = node[0]
            # Check if the node is a leaf
            if children_left[node_id] == children_right[node_id]:
                yield node
                continue

            # Build this node children, and add them at the beginning of the queue
            children = self._make_node_children(
                tree.tree_.feature,
                tree.tree_.threshold,
                children_left,
                children_right,
                node,
            )
            children.extend(queue)
            queue = children

    def _build_partitions(self, tree: DecisionTreeRegressor, samples: pd.DataFrame) -> list[HVSPartition]:
        """
        Build all the partitions using the given tree and samples

        :param tree: The decision tree to retrieve the partitions from
        :type tree: DecisionTreeRegressor
        :param samples:
            The samples used to train the decision tree
        :type samples: pd.DataFrame
        :return: A list of all the partitions retrieved from the tree
        :rtype: list[HVSPartition]
        """

        # Assign every sample to a leaf in the decision tree
        nodes_samples = {}
        for node, local_data in samples.groupby(tree.apply(samples[self.ordered_features])):
            nodes_samples[node] = local_data

        res = []
        for p in self._parse_tree(tree):
            node_id = p[0]
            bounds = p[1]

            new_partition = HVSPartition(
                bounds,
                nodes_samples[node_id],
                node_id,
                self.error_metric,
                self.objective,
                self.size_metric,
            )
            res.append(new_partition)

        # Sort the partitions by score
        res = sorted(res, key=lambda x: x.score, reverse=False)

        return res

    def _remove_extra_samples(
        self,
        n_samples: int,
        partitions: list[HVSPartition],
        partitions_samples: dict[int, int],
    ):
        # We must have exactly n_samples samples, so we remove any extra samples from the
        # partitions with the lowest
        # variance
        extra = sum(partitions_samples.values()) - n_samples
        if extra <= 0:
            return partitions_samples

        # Remove samples until we have exactly n_samples samples
        index = 0
        while extra > 0:
            node_index = partitions[index].node_id
            dif = min(extra, partitions_samples[node_index])
            extra -= dif
            partitions_samples[node_index] -= dif
            index += 1

        return partitions_samples

    def distribute_samples(self, partitions: list[HVSPartition], n_samples: int) -> list[tuple[int, HVSPartition]]:
        """
        Distribute a given number of samples to each partition,
        weighted using the partitions relative score compared to the sum of all the scores

        :param partitions: A list of the partitions to distribute the samples on
        :type partitions: list[HVSPartition]
        :param n_samples: The number of samples to distribute on the partitions
        :type n_samples: int
        :return: A list of tuple(int, HVSPartition), where the first element is the number of
            samples to take, and the second the corresponding partition
        :rtype: list[tuple[int, HVSPartition]]
        """

        scores_sum = sum(p.score for p in partitions)

        if scores_sum == 0:
            partitions_samples = {p.node_id: n_samples // len(partitions) for p in partitions}
        else:
            partitions_samples = {p.node_id: int(ceil(p.score / scores_sum * n_samples)) for p in partitions}

        partitions_samples = self._remove_extra_samples(n_samples, partitions, partitions_samples)
        return [(partitions_samples[p.node_id], p) for p in partitions]


class HVSampler(AdaptiveSampler):
    """Hierchical Variance Sampling (HVS) is an adaptive sampling strategy based on
    decision-tree partitionning of the sampling space

    The algorithm is described in the following paper:
    https://hal.science/hal-00952307 (DOI:10.1007/978-3-642-32820-6_11)

    It works by partitioning the space using a decision tree, and adding new samples based on the
    partitions size and
    variance. If the sampled function is multi-output, a decision tree is created for each
    output, and the new samples are split
    between each tree

    This class should be used in conjunction with the AdaptiveSamplingOrchestrator class for
    automated adaptive sampling
    If used directly, the user must handles the stopping criteria and the final dataset himself

    >>> features = {"x": [0, 5]}
    >>> # Define the function to sample. which MUST return a dataframe containing the original
    >>> # dataframe, and the sampled values as new columns
    >>> f = lambda df: pd.concat([df, df.apply(lambda x: x[0], axis=1)], axis=1)
    >>> sampler = HVSampler({"x": "int"}, features)
    >>> samples = sampler.sample(10, None, f)
    >>> # In this case, x = y for all samples
    >>> all(samples.iloc[:, 0] == samples.iloc[:, 1])
    True

    >>> print(list(samples.columns))
    ['x', 0]
    >>> # The dataframe contains the samples, that are correctly bounded according to the defined
    >>> # features
    >>> all(0 <= val <= 5 for val in samples.iloc[:, -1])
    True

    >>> # The sampler will throw on invalid parameters
    >>> samples = sampler.sample(-1, None, f)
    Traceback (most recent call last):
    ...
    mlkaps.sampling.sampler.SamplerError: n_samples must be a positive integer
    """

    def __init__(
        self,
        variables_types: dict | None = None,
        variables_values: dict | None = None,
        error_metric: str = "variance",
        min_samples_per_leaf: int = 15,
    ):
        """
        Create a new HVS sampler

        :param variables_types:
            A dictionary containing the types of the variables to sample.
            The keys must be the name of the variables, and the values must be one of
            ["int", "float", "categorical", "Boolean"].
            If None, the variables must be set later using the set_variables method.
            Defaults to None
        :type variables_types: dict | None, optional
        :param variables_values: A dictionary containing the values of the variables to sample.
            The keys must be the name of the variables, and the values must be a tuple
            (min, max) containing the bounds of the variable for numerical variables, or a list
            containing the possible values for categorical variables.
            If None, the variables must be set later using the set_variables method.
            Defaults to None
        :type variables_values: dict | None, optional
        :param error_metric:
            name of the error metric to use, one of ["variance", "cov"], defaults to "variance"
        :type error_metric: str, optional
        :param min_samples_per_leaf:
            The minimum number of samples per leaf in the decision tree, defaults to 15
        :type min_samples_per_leaf: int, optional
        """

        # If true, then some of the variables must be mapped from categorical
        # to numeric

        self.has_mapped_features = False
        self.numerical_features = None
        self.min_samples_per_leaf = min_samples_per_leaf

        super().__init__(variables_types, variables_values)

        self.variances = None
        self.errors = None
        self.final_partitions: list[HVSPartition] = None
        self.error_metric = error_metric

    def reset(self):
        """
        Reset and initialize the sampler internal state before starting a new sampling session
        """

        self.errors = None
        self.final_partitions = None

    def set_variables(self, variables_types: dict, variables_values: dict, mask: list = None):
        super().set_variables(variables_types, variables_values, mask)
        self._build_bounded_features()

    def _build_bounded_features(self):
        """
        Check if any of the variables is of non-float type, and if so,
        create map to numeric values. This also includes integers which are casted to float
        """

        if self.variables_values is None or self.variables_types is None:
            self.has_mapped_features = False
            return

        # Check whether any of the features non-float
        # If true, then we must convert the features to float values
        if not any(v in ["Categorical", "Boolean", "int"] for v in self.variables_types.values()):
            self.has_mapped_features = False
            self.numerical_features = self.variables_values
            return

        self.has_mapped_features = True

        self.numerical_features = convert_variables_bounds_to_numeric(self.variables_types, self.variables_values)

    def dump(self, output_directory: Path):
        """
        Output the median errors to a csv file in the given directory as
        "hvs_errors.csv"

        :param output_directory: The path where to save the error
        :type output_directory: pathlib.Path
        """

        if self.errors is not None:
            self.errors.to_csv(output_directory / "hvs_errors.csv")

    def _verify_arguments(
        self,
        n_samples: int,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
        features: dict,
    ):
        """
        Verify the arguments passed to the sampled

        :raise SamplerError: if any of the parameter is invalid
        """

        if not isinstance(n_samples, int) or n_samples < 0:
            raise SamplerError("n_samples must be a positive integer")

        if not isinstance(execution_func, Callable):
            raise SamplerError("execution_func must be a callable function")

        if not isinstance(features, dict):
            raise SamplerError("features must be a dictionary")

        if len(features) < 1:
            raise SamplerError("There must be at least 1 feature to sample")

    def _lhs_bootstrap(self, execution_func: Callable[[pd.DataFrame], pd.DataFrame], n_samples: int) -> pd.DataFrame:
        """
        Bootstrap the sampling process using LHS

        :param execution_func: The execution function to evaluate the samples
        :type execution_func: Callable[[pd.DataFrame], pd.DataFrame]
        :param n_samples: The number of samples to take using LHS
        :type n_samples: int
        :return: A list of samples
        :rtype: pd.DataFrame
        """

        # Use LHS to generate a starting dataset
        # Extract the features ranges as ndarray
        samples = LhsSampler(self.variables_types, self.variables_values)(n_samples)
        labels = execution_func(samples)

        return labels

    def sample(
        self,
        n_samples: int,
        samples: pd.DataFrame | None,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Samples new points from the given function, based on existing data
        If data is None, the sampler will bootstrap the dataset via LHS sampling using n_samples

        :param n_samples:
            The number of samples to generate, will be split between the objectives
            Must be >= 0
            If == 0, this function will return the input data and no sampling will be performed
        :type n_samples: int
        :param data:
            A list of already sampled points, or None
            If None, the sampler will bootstrap the dataset by using LHS sampling
        :type data: pandas.DataFrame
        :param execution_func:
            The callable function to sample
            Must accept a pd.DataFrame as input, and return a dataframe containing both the
            original dataframe, and the sampled values as new columns
        :type execution_function: Callable[[pandas.DataFrame], pandas.DataFrame]

        :return:
            A dataframe containing the extended dataset
        :rtype: pandas.DataFrame
        """

        if n_samples == 0:
            return samples

        self._verify_arguments(n_samples, execution_func, self.variables_values)

        # If no data is provided, we bootstrap the dataset
        if samples is None or len(samples) == 0:
            return self._lhs_bootstrap(execution_func, n_samples)

        # Ensure that we use numerical data, and map appropriately the categorical variables
        mapped_variables = self._maybe_map_variables(samples)

        new_data, errors = self._sample_for_all_objectives(n_samples, mapped_variables, execution_func)

        self.errors = pd.concat([self.errors, errors])
        samples = pd.concat([samples, new_data])

        return samples

    def _sample_for_all_objectives(
        self,
        n_samples: int,
        samples: pd.DataFrame,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a decision tree for each objective, and sample from each tree based on the variance

        :param n_sample: The number of samples to generate
        :type n_samples: int
        :param data: The available samples
        :type data: pandas.DataFrame
        :param execution_func: The function to sample
        :type execution_func: Callable[[pandas.DataFrame], pandas.DataFrame]

        :return: An extended dataset, and the errors for each objective
        :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
        """

        objectives = [c for c in samples.columns if c not in self.numerical_features]
        samples_per_objectives = int(ceil(n_samples / len(objectives)))

        new_points = []
        errors = []

        for obj in objectives:
            partitions, median_error = self.partition(
                samples,
                obj,
                samples_per_objectives,
                min_samples_per_leaf=self.min_samples_per_leaf,
            )

            # Keep track of the evolution of the variance
            errors.append(median_error)

            # Sample from each partition
            to_sample = self.sample_partitions(partitions)

            new_points.append(to_sample)

        new_points = pd.concat(new_points)
        new_points.reset_index(drop=True, inplace=True)

        errors = pd.DataFrame(np.array(errors)[None], columns=objectives)

        labelled_results = execution_func(new_points)

        return labelled_results, errors

    def partition(
        self,
        samples: pd.DataFrame,
        objective: str,
        n_samples: int,
        split_on: list[str] | None = None,
        min_samples_per_leaf: int = 15,
    ) -> tuple[list[HVSPartition], float]:
        """
        Generate the partitions for a given set of samples

        :param samples: The samples to build the partitions with
        :type samples: pd.DataFrame
        :param objective: The name of the feature in the samples to consider as the objective
        :type objective: str
        :param n_samples: The number of samples to distribute over the partitions
        :type n_samples: int
        :param split_on:
            The features to consider when building the decision tree, defaults to None.
            If None, takes the features defined in the constructor or using set_variables
        :type split_on: list[str] | None, optional
        :param min_samples_per_leaf:
            The minimum number of samples per leaf in the decision tree, defaults to 15
        :type min_samples_per_leaf: int, optional
        :return: A list of partitions, and the median error
        :rtype: tuple[list[HVSPartition], float]
        """

        if split_on is None:
            # If the splitting features were not given, split on all features
            split_on = self.numerical_features

        # Extract the input from the data, and ensure the ordering is alpha-numerical
        partitionner = HVSPartitionner(
            split_on,
            objective,
            # FIXME: Currently, the only parameter is the min number of samples in the leaf
            # We should improve this down the line
            {"min_samples_leaf": min_samples_per_leaf},
            self.error_metric,
        )
        partitions = partitionner.partition(samples)
        self.final_partitions = partitions
        self.final_tree = partitionner.tree
        median_error = np.median([p.error for p in partitions])

        partitions = partitionner.distribute_samples(partitions, n_samples)

        return partitions, median_error

    def sample_partitions(self, decorated_partitions: list[HVSPartition]) -> pd.DataFrame:
        """
        Samples each partition using LHS, and outputs a dataframe with all the samples
        The number of samples is given by the "n_samples" key of each partition
        """
        res = []
        for n_samples, partition in decorated_partitions:
            axes = partition.axes

            # Reorder the axes to match the order of the features
            ranges = np.array(list(axes.values()))
            sampler = Random(xlimits=ranges)
            samples = pd.DataFrame(sampler(n_samples), columns=axes.keys())

            res.append(samples)

        res = pd.concat(res)
        res = res.reindex(sorted(res.columns), axis=1)

        # Run the executor on the new samples
        # Map the new points back to the categorical space if necessary
        res = self._maybe_map_variables(res, reverse=True)
        return res

    def _maybe_map_variables(self, samples: pd.DataFrame, reverse: bool = False):

        if not self.has_mapped_features:
            return samples

        mapped_data = None
        if reverse:
            mapped_data = map_float_to_variables(samples, self.variables_types, self.variables_values)
        else:
            mapped_data = map_variables_to_numeric(samples, self.variables_types, self.variables_values)
        return mapped_data
