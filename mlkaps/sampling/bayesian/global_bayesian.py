"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pymoo.core.problem import Problem
from mlkaps.sampling.bayesian.kdtree import KDTree, KDTreeNode
from typing import Callable


class AcquisitionFunction:
    known_func = [
        "EI",
        "EI_size",
        "EI_rel",
        "EI_rel_size",
        "PI",
        "PI_size",
        "EI_logrel",
        "EI_logrel_size",
    ]

    def __init__(self, acq, maximize=False):
        self.acq = acq
        # If maximize is set to True, i.e higher objective values are better
        # We flip all operations to properly compute the EI
        self.maximize = maximize
        assert (
            acq in AcquisitionFunction.known_func
        ), f'The acquisition function "{acq}" is not implemented. Must be one of {AcquisitionFunction.known_func}'

    def __call__(self, x, pred, sigma):
        if self.acq == "EI":
            Z = self._compute_Z(x, pred, sigma)
            EI = (self._compute_diff(x, pred) * norm.cdf(Z)) + sigma * norm.pdf(Z)
            return -np.maximum(0, EI.values)
        elif self.acq == "EI_size":
            Z = self._compute_Z(x, pred, sigma)
            EI = (self._compute_diff(x, pred) * norm.cdf(Z)) + sigma * norm.pdf(Z)
            return -np.maximum(0, EI.values) * x["size"].values
        elif self.acq == "EI_rel":
            Z = self._compute_Z(x, pred, sigma)
            SU = self._compute_su(x, pred)
            EI = (SU * norm.cdf(Z)) + (sigma / x["best_o"]) * norm.pdf(Z)
            return -np.maximum(0, EI.values)
        elif self.acq == "EI_rel_size":
            Z = self._compute_Z(x, pred, sigma)
            SU = self._compute_su(x, pred)
            EI = (SU * norm.cdf(Z)) + (sigma / x["best_o"]) * norm.pdf(Z)
            return -np.maximum(0, EI.values) * x["size"].values
        elif self.acq == "PI":
            Z = self._compute_Z(x, pred, sigma)
            PI = norm.cdf(Z)
            return -PI
        elif self.acq == "PI_size":
            Z = self._compute_Z(x, pred, sigma)
            PI = norm.cdf(Z)
            return -PI * x["size"].values
        else:
            raise NotImplementedError()

    def _compute_su(self, x, pred):
        epsilon = 1e-8
        if self.maximize:
            return (pred - x["best_o"]) / (x["best_o"] + epsilon)
        else:
            return (x["best_o"] - pred) / (x["best_o"] + epsilon)

    def _compute_diff(self, x, pred):
        if self.maximize:
            return pred - x["best_o"]
        else:
            return x["best_o"] - pred

    def _compute_Z(self, x, pred, sigma):
        # Small epsilon to avoid division by zero

        epsilon = 1e-8
        if self.maximize:
            # We want to maximize the objective
            Z = (pred - x["best_o"]) / (sigma + epsilon)
        else:
            # We want to minimize the object, we need to flip the subtraction
            Z = (x["best_o"] - pred) / (sigma + epsilon)

        return Z

    def single_partition(self, x, pred, sigma):
        # This is a helper function to evaluate the acquisition function on a single partition
        # Which means the size is constant and doesn't matter
        tmp = x.copy()
        tmp["size"] = 1.0
        return self(tmp, pred, sigma)


class BayesianOptimizationProblem(Problem):

    def __init__(self, feature_values, feature_types, model, acq: AcquisitionFunction, **kwargs):
        self.model = model
        self.feature_values = feature_values
        self.acq = acq
        self.ordering = sorted(list(feature_values.keys()))

        xl = [self.feature_values[p][0] for p in self.ordering]
        xu = [self.feature_values[p][1] for p in self.ordering]
        super().__init__(
            n_var=len(self.kernel_parameters),
            n_obj=1,
            n_constr=0,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # First, for each X, finds the corresponding partition and the current best value
        x = pd.DataFrame(x, columns=self.kernel_parameters)
        r = self.fastassign(x)

        x["best_o"] = r[:, 0]
        x["size"] = r[:, 1]

        pred, variance = self.model.predict(x[self.kernel_parameters])
        sigma = np.sqrt(variance)

        out["F"] = self.acq(x, pred, sigma)

    def _fastassign(self, x):
        ids = self.tree.predict(x)
        r = np.ndarray(shape=(x.shape[0], 2))

        for i, id in enumerate(ids):
            # We need to compute the best value and the size of the partition
            # We can use the KDTreeNode to do that
            node = self.tree.nodes[id]
            r[i, 0] = node.data["best_o"]
            r[i, 1] = node.volume


class KDTreeSplitter:
    def __init__(self, objective: str, directions: dict[str], partitionner: KDTree, model):
        self.input_features = partitionner.ordering
        self.directions = directions
        self.partitionner = partitionner
        self.objective = objective
        self.model = model

    def variance_split(self, samples, max_depth=5, root=0):
        # Fit a sklearn decision tree for variance reduction
        # And then rebuild the corresponding KDTree in our own format
        # This is simpler and faster than reimplementing our own
        # Split function
        from sklearn.tree import DecisionTreeRegressor

        X = samples[self.input_features]
        y = samples[self.objective]
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X, y)

        sk_tree = tree.tree_

        stack = [(0, root)]

        while stack:
            sk_id, custom_node = stack.pop()

            if sk_tree.children_left[sk_id] == -1:  # Leaf node
                continue

            feature_idx = sk_tree.feature[sk_id]
            threshold = sk_tree.threshold[sk_id]
            feature_name = self.input_features[feature_idx]

            left_child, right_child = self.partitionner.split(custom_node, feature_name, threshold)

            stack.append((sk_tree.children_right[sk_id], right_child.id))
            stack.append((sk_tree.children_left[sk_id], left_child.id))

    def _should_split_cv(self, model, next):
        partition = self.partitionner.predict(next)[0]
        partition = self.partitionner.nodes[partition]

        from mlkaps.sampling.generic_bounded_sampler import RandomSampler
        sampler = RandomSampler(self.input_features, self.feature_values, self.partitionner.ordering)
        samples = sampler(2048)
        design_parameters = np.tile(partition.data["parameters"].values, (samples.shape[0], 1))
        design_parameters = pd.DataFrame(design_parameters, columns=partition.data["parameters"].index)
        samples = pd.concat([samples, design_parameters], axis=1)

        pred, _ = model.predict(samples)

        mean = pred.mean()
        std = pred.std()
        cv = std / mean
        return cv < 0.1, partition
        

    def cv_split(self, model, samples, next):
        cv_low, partition = self._should_split_cv(model, next)

        if cv_low:
            # If we decide not to split the partition
            # We need to check whether the new sample is better than the current best
            if self._is_better(next[self.objective].values[0], partition.value["best_o"]):
                partition.value["best_o"] = next[self.objective].values[0]
                partition.value["parameters"] = next[self.input_features].values[0]
        else:
            # We take all the samples in the partition
            ids = self.partitionner.predict(samples)
            csamples = samples[ids == partition.id]
            # We split the partition using the variance split
            self.variance_split(csamples, max_depth=1, root=partition.id)

            lid, rid = self.partitionner.lefts[partition.id], self.partitionner.rights[partition.id]

            split_axis = self.partitionner.split_axis[partition.id]
            threshold = self.partitionner.nodthresholdes[partition.id]
            left = csamples.iloc[:, split_axis] <= threshold
            csamples = [csamples[left], csamples[~left]]

            for node, nsamples in zip([self.partitionner.nodes[lid], self.partitionner.nodes[rid]], csamples):
                if len(nsamples) < 20:
                    tmp = self.random_samples(node, 20 - len(nsamples))
                    samples = pd.concat([samples, tmp])
                    nsamples = pd.concat([nsamples, tmp])

                idx = self._idx_of_optimum(nsamples)
                node.data = {"best_o": nsamples.loc[idx][self.objective], "parameters": nsamples.loc[idx][self.input_features]}

        return samples

    def _idx_of_optimum(self, samples: pd.DataFrame) -> int:
        """Returns the index of the best samples according to the objective

        :param samples: The samples to check
        :type samples: pd.DataFrame
        :raises NotImplementedError: Raised when the direction is not known
        :return: The index of the best sample
        :rtype: int
        """
        if self.directions[self.objective] == "minimize":
            return samples[self.objective].idxmin()
        elif self.directions[self.objective] == "maximize":
            return samples[self.objective].idxmax()
        else:
            raise NotImplementedError(f"Unknown direction {self.directions[self.objective]}")

    def _is_better(self, v1: float, v2: float) -> bool:
        """Check whether v1 is better than v2 according to the objective direction

        If we are maximizing, we want to check if v1 > v2
        If we are minimizing, we want to check if v1 < v2

        :param v1: The value to check for
        :type v1: float
        :param v2: The value to check against
        :type v2: float
        :return: True if v1 is better than v2
        :rtype: bool
        """
        if self.directions[self.objective] == "minimize":
            return v1 < v2
        elif self.directions[self.objective]  == "maximize":
            return v1 > v2


class BayesianSampler:
    def __init__(
        self,
        kernel_harness: Callable[[pd.DataFrame], pd.DataFrame],
        input_features: list[str],
        feature_values: dict[str, list[float]],
        feature_types: dict[str, str],
        directions: dict[str, str],
        acq: AcquisitionFunction,
        bootstrap_ratio=0.1,
    ):
        self.input_features = input_features
        self.design_parameters = [p for p in feature_values.keys() if p not in input_features]
        self.feature_values = feature_values
        self.feature_types = feature_types
        self.acq = acq
        self.kernel = kernel_harness

        assert len(directions) == 1, "Bayesian Sampler only supports single objective optimization in current implementation"
        self.directions = directions

        self.bootstrap_ratio = bootstrap_ratio

    def _boostrap(self, nsamples):
        from mlkaps.sampling.generic_bounded_sampler import LhsSampler

        sampler = LhsSampler(self.feature_types, self.feature_values)
        samples = sampler(int(nsamples * self.bootstrap_ratio))
        samples = self.kernel(samples)

        return samples

    def _fit_model(self, samples):
        from mlkaps.modeling.quantile_variance_estimator import QuantileLGBMMixture

        model = QuantileLGBMMixture(hb_alpha=0.975)
        objective = list(self.directions.keys())[0]
        model.fit(samples[self.design_parameters], samples[objective])

        return model

    def _init_partitionner(self, samples, model):
        spatial_features = {k: v for k, v in self.feature_values.items() if k in self.input_features}
        partitionner = KDTree(spatial_features, self.feature_types)

        objective = list(self.directions.keys())[0]
        # Split the initial tree a few time
        splitter = KDTreeSplitter(objective, self.directions, partitionner, model)
        splitter.variance_split(samples)

        ids = partitionner.predict(samples)
        for id, samples in samples.groupby(ids):
            node = partitionner.nodes[id]
            idxmin = samples[objective].idxmin()
            node.data = {"best_o": samples.loc[idxmin][objective], "parameters": samples.loc[idxmin][self.design_parameters]}

        return partitionner

    def _iterate(self, samples, nsamples):
        model = self._fit_model(samples)
        partitionner = self._init_partitionner(samples, model)
        splitter = self._build_splitter()

        problem, minimizer = self._create_problem(model, partitionner)
        niter = 0

        while len(samples) < nsamples:
            next = minimizer(problem, samples)
            next = self.kernel(next)

            samples = pd.concat([samples, next])
            samples.reset_index(drop=True, inplace=True)

            samples = splitter(model, samples, next)

            if niter % 10 == 0:
                model = self._fit_model(samples)
                problem.model = model

            if self.plot and niter % 50 == 0:
                self._plot(samples, model, partitionner)

        return samples

    def __call__(self, samples, nsamples):
        if samples is None:
            samples = self._boostrap(nsamples)

        samples = self._iterate(samples, nsamples)
        return samples
