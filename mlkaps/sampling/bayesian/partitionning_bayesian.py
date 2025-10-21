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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlkaps.sampling.bayesian.kdtree import KDTree, KDTreeNode
from mlkaps.sampling.generic_bounded_sampler import RandomSampler
from typing import Callable
from enum import StrEnum
from mlkaps.modeling.encoding import encode_dataframe
from tqdm import tqdm
import copy
from pathlib import Path
import logging

logger = logging.getLogger("GlobalBayesian")


def sample_in_node(feature_values, feature_types: dict[str, str], node: KDTreeNode, count: int) -> pd.DataFrame:
    """
    Generate uniformly-distributed samples inside the bound of the provided KDTreeNode

    :param feature_types: The type of each variables
    :type feature_types: dict[str, str]
    :param node: The KDTreeNode to sample from
    :type node: KDTreeNode
    :param count: The number of samples to take
    :type count: int
    :return: A pandas DataFrame containing all the new samples
    :rtype: pd.DataFrame
    """
    bounds = copy.deepcopy(feature_values)
    for k, v in node.bounds.items():
        bounds[k] = v

    sampler = RandomSampler(feature_types, bounds)
    samples = sampler(count)

    return samples


def find_samples_in_node(samples: pd.DataFrame, node: KDTreeNode) -> pd.DataFrame:
    # This function will fin all samples inside the node bounds
    # And then interchange all the design parameters
    # To find the combination that maximizes the mean objective value inside the node
    mask = pd.Series(True, index=samples.index)
    for param, (lb, ub) in node.bounds.items():
        mask &= (samples[param] >= lb) & (samples[param] <= ub)
    return samples.loc[mask]


def best_in_node(direction, samples, node, model):
    # This function will fin all samples inside the node bounds
    # And then interchange all the design parameters
    # To find the combination that maximizes the mean objective value inside the node
    inside_samples = find_samples_in_node(samples, node)

    if inside_samples.empty:
        raise ValueError("No samples found inside the node bounds.")

    input_params = list(node.bounds.keys())
    design_params = [col for col in samples.columns if col not in input_params]

    inputs = inside_samples[input_params].reset_index(drop=True)
    designs = inside_samples[design_params].reset_index(drop=True)

    best_median_perf = None
    best_design = None

    unique_designs = designs.drop_duplicates().reset_index(drop=True)

    for i, design_row in unique_designs.iterrows():
        combined = inputs.copy()
        for col in design_params:
            combined[col] = design_row[col]

        preds = model.predict(combined)[0]
        mean_perf = np.median(preds)

        if (
            (best_median_perf is None)
            or (mean_perf > best_median_perf and direction == "maximize")
            or (mean_perf < best_median_perf and direction == "minimize")
        ):
            best_median_perf = mean_perf
            best_design = design_row

    node.data = {"best_o": best_median_perf, "parameters": best_design}

    return best_design


class AcquisitionFunctions(StrEnum):
    EI = "EI"
    EI_size = "EI_size"
    EI_rel = "EI_rel"
    EI_rel_size = "EI_rel_size"
    PI = "PI"
    PI_size = "PI_size"
    EI_logrel = "EI_logrel"
    EI_logrel_size = "EI_logrel_size"


class AcquisitionFunction:
    def __init__(self, acq: str | AcquisitionFunctions, maximize=False):
        self.acq = AcquisitionFunctions(acq)
        # If maximize is set to True, i.e higher objective values are better
        # We flip all operations to properly compute the EI
        self.maximize = maximize

    def __call__(self, x, pred, sigma):
        if self.acq == AcquisitionFunctions.EI:
            z = self._compute_z(x, pred, sigma)
            ei = (self._compute_diff(x, pred) * norm.cdf(z)) + sigma * norm.pdf(z)
            return -np.maximum(0, ei.values)
        elif self.acq == AcquisitionFunctions.EI_size:
            z = self._compute_z(x, pred, sigma)
            ei = (self._compute_diff(x, pred) * norm.cdf(z)) + sigma * norm.pdf(z)
            return -np.maximum(0, ei.values) * x["size"].values
        elif self.acq == AcquisitionFunctions.EI_rel:
            z = self._compute_z(x, pred, sigma)
            su = self._compute_su(x, pred)
            ei = (su * norm.cdf(z)) + (sigma / x["best_o"]) * norm.pdf(z)
            return -np.maximum(0, ei.values)
        elif self.acq == AcquisitionFunctions.EI_rel_size:
            z = self._compute_z(x, pred, sigma)
            su = self._compute_su(x, pred)
            ei = (su * norm.cdf(z)) + (sigma / x["best_o"]) * norm.pdf(z)
            return -np.maximum(0, ei.values) * x["size"].values
        elif self.acq == AcquisitionFunctions.PI:
            z = self._compute_z(x, pred, sigma)
            pi = norm.cdf(z)
            return -pi
        elif self.acq == AcquisitionFunctions.PI_size:
            z = self._compute_z(x, pred, sigma)
            pi = norm.cdf(z)
            return -pi * x["size"].values
        else:
            raise NotImplementedError(f"Acquisition Function {self.acq=} is not implemented !")

    def ei_components(self, x, pred, sigma):
        match self.acq:
            case AcquisitionFunctions.EI | AcquisitionFunctions.EI_size:
                z = self._compute_z(x, pred, sigma)
                exploitation = self._compute_diff(x, pred) * norm.cdf(z)
                exploration = sigma * norm.pdf(z)[0]
                return {
                    "ptarget": pred[0],
                    "pbest_o": x["best_o"][0],
                    "z": z[0],
                    "exploitation": exploitation[0],
                    "exploration": exploration[0],
                    "total": exploitation[0] + exploration[0],
                    "psigma": sigma[0],
                }
            case AcquisitionFunctions.EI_rel | AcquisitionFunctions.EI_rel_size:
                z = self._compute_z(x, pred, sigma)
                su = self._compute_su(x, pred)
                exploitation = su * norm.cdf(z)
                exploration = (sigma / x["best_o"]) * norm.pdf(z)
                return {
                    "pbest_o": x["best_o"][0],
                    "ptarget": pred[0],
                    "psu": su[0],
                    "z": z[0],
                    "exploitation": exploitation[0],
                    "exploration": exploration[0],
                    "total": exploitation[0] + exploration[0],
                    "psigma": sigma[0],
                }
            case AcquisitionFunctions.PI | AcquisitionFunctions.PI_size:
                z = self._compute_z(x, pred, sigma)
                pi = norm.cdf(z)
                return {"pi": pi[0], "z": z[0], "psigma": sigma[0], "ptarget": pred[0], "pbest_o": x["best_o"][0]}
            case _:
                raise NotImplementedError(f"Acquisition Function {self.acq=} is not implemented !")

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

    def _compute_z(self, x, pred, sigma):
        # Small epsilon to avoid division by zero

        epsilon = 1e-8
        if self.maximize:
            # We want to maximize the objective
            z = (pred - x["best_o"]) / (sigma + epsilon)
        else:
            # We want to minimize the object, we need to flip the subtraction
            z = (x["best_o"] - pred) / (sigma + epsilon)

        return z

    def single_partition(self, x, pred, sigma):
        # This is a helper function to evaluate the acquisition function on a single partition
        # Which means the size is constant and doesn't matter
        tmp = x.copy()
        tmp["size"] = 1.0
        return self(tmp, pred, sigma)


class BayesianOptimizationProblem(Problem):

    def __init__(
        self,
        feature_values: dict[str, list],
        feature_types: dict[str, str],
        input_features: list[str],
        model,
        partitioner: KDTree,
        acq: AcquisitionFunction,
        **kwargs,
    ):
        self.model = model
        self.feature_values = feature_values
        self.feature_types = feature_types
        self.input_features = input_features
        self.partitioner = partitioner
        self.acq = acq
        self.ordering = sorted(list(feature_values.keys()))

        mixed_vars = self._define_vars()
        print(mixed_vars)
        super().__init__(
            n_obj=1,
            n_constr=0,
            vars=mixed_vars,
            **kwargs,
        )

    def _define_vars(self):
        from pymoo.core.variable import Choice, Real, Binary, Integer

        # To define a mixed precision problem, we need to define each
        # variable, and their respective bound
        feature_values = self.feature_values

        mixed_vars = {}
        for name, parameter_type in self.feature_types.items():
            match parameter_type:
                case "float":
                    pymoo_var = Real(bounds=feature_values[name].get_sampling_bounds())
                case "int":
                    pymoo_var = Integer(bounds=feature_values[name].get_sampling_bounds())
                case "Boolean":
                    pymoo_var = Binary()
                case "Categorical":
                    pymoo_var = Choice(options=feature_values[name].values)
                case _:
                    raise ValueError(f"Unexpected variable type for '{name}' ('{parameter_type}')")
            mixed_vars[name] = pymoo_var
        return mixed_vars

    def _evaluate(self, x, out, *args, **kwargs):
        # First, for each X, finds the corresponding partition and the current best value
        # As we are in mixed variable mode, we receive an np.array of python dicts
        # We need to cast this to a list of dicts or pandas will be confused when building the DataFrame
        self.model.mean_only = False  # We want to predict the variance as well
        x = pd.DataFrame(list(x))
        r = self._fast_assign(x)

        x["best_o"] = r["target"]
        x["size"] = r["size"]

        pred, variance = self.model.predict(x)
        sigma = np.sqrt(variance)

        out["F"] = self.acq(x, pred, sigma)

    def _fast_assign(self, x):
        ids = self.partitioner.predict(x)
        copy = x[self.input_features].copy()

        parameters = []
        sizes = []

        for id in ids:
            node = self.partitioner.nodes[id]
            parameters.append(node.data["parameters"])
            sizes.append(node.volume)

        param_df = pd.concat(parameters, axis=1).T.reset_index(drop=True)
        copy[param_df.columns] = param_df

        copy["target"] = self.model.predict(copy)[0]
        copy["size"] = sizes

        return copy


class KDTreeSplitter:
    def __init__(
        self,
        kernel: Callable[[pd.DataFrame], pd.DataFrame],
        feature_values: dict[str, list],
        feature_types: dict[str, str],
        objective: str,
        directions: dict[str, str],
        partitioner: KDTree,
        model,
    ):

        self.kernel = kernel
        self.feature_values = feature_values
        self.feature_types = feature_types
        self.input_features = partitioner.ordering
        self.design_parameters = [p for p in feature_values.keys() if p not in self.input_features]
        self.directions = directions
        self.partitioner = partitioner
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
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=20)
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

            left_child, right_child = self.partitioner.split(custom_node, feature_name, threshold)

            stack.append((sk_tree.children_right[sk_id], right_child.id))
            stack.append((sk_tree.children_left[sk_id], left_child.id))

    def _should_split_cv(self, model, next):
        partition = self.partitioner.predict(next)[0]
        partition = self.partitioner.nodes[partition]

        from mlkaps.sampling.generic_bounded_sampler import RandomSampler

        sampler = RandomSampler(self.feature_types, self.feature_values, self.input_features)
        samples = sampler(2048)
        design_parameters = np.tile(partition.data["parameters"].values, (samples.shape[0], 1))
        design_parameters = pd.DataFrame(design_parameters, columns=partition.data["parameters"].index)
        samples = pd.concat([samples, design_parameters], axis=1)

        pred, _ = model.predict(samples)

        mean = pred.mean()
        std = pred.std()
        cv = std / mean
        return cv < 0.1, partition

    def _check_new_configuration(self, samples, node, model, new_configuration):
        samples = find_samples_in_node(samples, node)

        for col in node.data["parameters"].index:
            samples[col] = node.data["parameters"][col]

        model.mean_only = True
        pred = model.predict(samples)

        if isinstance(new_configuration, pd.DataFrame):
            if len(new_configuration) != 1:
                raise ValueError("New configuration should be a single row DataFrame")
            new_configuration = new_configuration.iloc[0]

        for col in new_configuration.index:
            if col in self.input_features:
                continue
            samples[col] = new_configuration[col]

        new_preds = model.predict(samples)
        return samples, pred, new_preds

    def _happy_unhappy_split(
        self, direction, samples, node, model, new_configuration, agreement_threshold=0.95, separability_threshold=0.9
    ) -> tuple[str, pd.Series, pd.Series, DecisionTreeClassifier | None]:
        # Here, we should cancel splits if there's isn't enough samples in the node
        snode = find_samples_in_node(samples, node)
        if snode.shape[0] < 40:
            return "NotEnoughData", pd.Series(dtype=bool), pd.Series(dtype=float), None

        random_samples_in_node = sample_in_node(self.feature_values, self.feature_types, node, 1000)
        random_samples_in_node.drop_duplicates(inplace=True)

        _, pred, new_preds = self._check_new_configuration(random_samples_in_node, node, model, new_configuration)

        # Now, we can determine which samples are happy or unhappy about the new configuration
        if direction == "maximize":
            happy = pred < new_preds
        else:
            happy = pred > new_preds

        happy_ratio = happy.mean()
        if happy_ratio >= agreement_threshold:
            return ("update", happy, new_preds, None)
        elif happy_ratio < (1.0 - agreement_threshold):
            # If nobody likes the new configuration, we keep the old one
            # And we do not split the node
            return "keep", happy, new_preds, None

        param_cols = list(node.bounds.keys())
        X = random_samples_in_node[param_cols].values
        y = happy.astype(int)

        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y)
        y_pred = stump.predict(X)
        acc = accuracy_score(y, y_pred)

        # We should check how many samples fall in each partition
        # And stop if if we have too few samples in each partition
        preds = stump.predict(snode[param_cols])
        left = np.sum(preds == 0)
        right = np.sum(preds == 1)

        if left < 20 or right < 20:
            return "NotEnoughData", happy, new_preds, None

        if acc >= separability_threshold:
            return "split", happy, new_preds, stump
        else:
            return "keep", happy, new_preds, None

    def cv_split(self, model, samples, new_sample):
        partition = self.partitioner.predict(new_sample)[0]
        partition = self.partitioner.nodes[partition]

        decision, happy, new_preds, stump = self._happy_unhappy_split(
            self.directions[self.objective],
            samples,
            partition,
            model,
            new_sample,
            agreement_threshold=0.95,
            separability_threshold=0.80,
        )

        if decision in ["keep", "NotEnoughData"]:
            if len(happy) == 0:
                count = np.nan
            else:
                count = np.sum(happy) / len(happy)
            return samples, {"decision": decision, "happy": count}
        elif decision == "update":
            best_in_node(self.directions[self.objective], samples, partition, model)
            return samples, {"decision": decision, "happy": np.sum(happy) / len(happy)}
        elif decision == "split":
            split_axis = stump.tree_.feature[0]
            threshold = stump.tree_.threshold[0]
            # We update the partitioner with the new split
            lid, rid = self.partitioner.split(partition.id, self.input_features[split_axis], threshold)

            for node in [lid, rid]:
                csamples = find_samples_in_node(samples, node)
                best_in_node(self.directions[self.objective], csamples, node, model)
        else:
            raise ValueError(f"Unknown decision {decision} for the KDTree splitter")

        return samples, {"decision": decision, "happy": np.sum(happy) / len(happy)}

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
            raise ValueError(f"Unknown direction {self.directions[self.objective]}")

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
        elif self.directions[self.objective] == "maximize":
            return v1 > v2
        else:
            raise ValueError(f"Unknown direction {self.directions[self.objective]}")


class PartitionningBayesianSampler:
    def __init__(
        self,
        kernel: Callable[[pd.DataFrame], pd.DataFrame],
        input_features: list[str],
        feature_values: dict[str],
        feature_types: dict[str, str],
        directions: dict[str, str],
        acq: str | AcquisitionFunction,
        bootstrap_ratio=0.1,
        output_dir: str | Path | None = None,
        **kwargs,
    ):
        """Build a new Global Bayesian sampler using the KD-Tree partitionning strategy.

        :param kernel: The black box kernel to sample from.
        :type kernel: Callable[[pd.DataFrame], pd.DataFrame]
        :param input_features: A list of features name to treat as input features.
        :type input_features: list[str]
        :param feature_values: A dict mapping each feature to its range or possible values for categorial features.
            Should be either dict[key] = [low, high] or dict[key] = ["a", "b", "c"] (For categorical).
        :type feature_values: dict[str, list[float]]
        :param feature_types: A dict mapping each feature to its type.
            Should be dict[key] = one of ["float", "int", "bool", "Categorical"].
        :type feature_types: dict[str, str]
        :param directions: A dict containing the directions of each objective.
            Should be dict[objective] = one of ["minimize", "maximize"].
            Note that only single objective optimization is currently supported.
        :type directions: dict[str, str]
        :param acq: The acquisition function to use for the bayesian algorithm.
            Should be one of AcquisitionFunctions.
        :type acq: str | AcquisitionFunction
        :param bootstrap_ratio: The ratio of the total number of samples to use for bootstrapping, defaults to 0.1.
            Current implementation uses Latin Hypercube Sampling during bootstrap.
        :type bootstrap_ratio: float, optional
        :param output_dir: The path the sampler should output to if needed
            This path is used for dumping and checkpointing the collected samples
        :type output_dir: str | Path | None
        :raises ValueError: Raised if any of the parameters is incorrect.
        """

        if not feature_values:
            raise ValueError("Cannot build sampler with no features to sample")

        if not feature_types:
            raise ValueError("Cannot build sampler with no feature types")

        missing_inputs = [f for f in input_features if f not in feature_values]
        if missing_inputs:
            raise ValueError(f"Input features missing in feature values: {missing_inputs}")

        missing_types = [f for f in feature_values if f not in feature_types]
        if missing_types:
            raise ValueError(f"Feature types missing for features: {missing_types}")

        if len(directions) != 1:
            raise ValueError("Only single-objective optimization is supported")

        unsupported_directions = [v for v in directions.values() if v not in {"maximize", "minimize"}]
        if unsupported_directions:
            raise ValueError(f"Unsupported optimization directions: {unsupported_directions}")

        if not (0 < bootstrap_ratio < 1.0):
            raise ValueError(f"bootstrap_ratio must be in (0, 1), got {bootstrap_ratio}")

        self.input_features = copy.deepcopy(input_features)
        self.design_parameters = [p for p in feature_values if p not in input_features]
        self.feature_values = copy.deepcopy(feature_values)
        self.feature_types = copy.deepcopy(feature_types)
        self.ordering = sorted(feature_values)

        if isinstance(acq, str):
            self.acq = AcquisitionFunction(acq)
        elif not isinstance(acq, AcquisitionFunction):
            raise TypeError(f"Expected str or AcquisitionFunction, got {type(acq)}")
        else:
            self.acq = acq

        # TODO: For now we only support a single objective, so this tricks works
        # In the future, we should update this code !
        self.acq.maximize = list(directions.values())[0] == "maximize"

        self.kernel = kernel
        self.directions = copy.deepcopy(directions)
        self.bootstrap_ratio = bootstrap_ratio
        self.output_dir = output_dir
        if output_dir is not None:
            self.output_dir.mkdir(exist_ok=True, parents=True)

        self.alpha = kwargs.pop("alpha", 0.841)

        if not (0.5 < self.alpha < 1.0):
            raise ValueError(
                f"Invalid value for alpha (0.5 < {self.alpha=} < 1.0)\n"
                f"When minimizing, we will automatically invert this value if needed."
            )

        # FIXME: Quick hack because we only support single objective optimization for now
        if list(self.directions.values())[0] == "minimize":
            # When we are minimizing, we don't care about predicting the upper quantile(s)
            # Because this would capture the tail/worst cases of the distribution
            # Which is not what we want when minimizing
            # So we invert alpha to capture the lower quantiles instead
            # Note: Technically, we probably should adjust alpha depending on the underlying conditional distribution
            self.alpha = 1.0 - self.alpha

        self.dump = kwargs.pop("dump", False)
        if not isinstance(self.dump, (int, bool)):
            raise ValueError(f"Dump should be an integer (dump interval) or False, received {self.dump}")

        if self.dump is False and self.output_dir is None:
            raise ValueError("Cannot dump if no output_dir is provided")

        self.initial_depth = kwargs.pop("initial_depth", 5)
        if self.initial_depth <= 0:
            raise ValueError(f"Initial depth should be >= 1, received {self.initial_depth}")

        if len(kwargs) != 0:
            raise ValueError(f"Unknown parameters: {kwargs}")

    def _boostrap(self, nsamples: int):
        from mlkaps.sampling.generic_bounded_sampler import LhsSampler

        assert nsamples > 0, "Number of samples cannot be negative"

        nbootstrap = int(nsamples * self.bootstrap_ratio)
        if nbootstrap <= 0:
            raise ValueError(
                f"Could not bootstrap, at least 1 sample is required, "
                f"requested ({nsamples=}x{self.bootstrap_ratio=}={nbootstrap})"
            )
        sampler = LhsSampler(self.feature_types, self.feature_values)

        samples = sampler(int(nsamples * self.bootstrap_ratio))
        samples = self.kernel(samples)

        return samples

    def _fit_model(self, samples: pd.DataFrame):
        from mlkaps.modeling.iqr_variance_estimator import IQRVarianceEstimator

        assert samples is not None and len(samples) > 0, f"Cannot fit model on None or empty dataframe {samples}"

        samples = encode_dataframe(self.feature_types, samples)
        features = list(self.feature_values.keys())

        model = IQRVarianceEstimator(alpha=self.alpha, method="forced_symmetry")
        objective = list(self.directions.keys())[0]
        model.fit(samples[features], samples[objective])

        return model

    def _dump(self, partitioner: KDTree, samples: pd.DataFrame):
        import pickle

        output_dir = self.output_dir / f"bayesian_dumps/{len(samples)}"
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / "partitioner.pkl", "wb") as f:
            pickle.dump(partitioner, f)

        samples.to_csv(output_dir / "samples.csv", index=False)

    def _build_minimizer(self, model, partitioner: KDTree) -> tuple[BayesianOptimizationProblem, Callable]:
        from pymoo.termination import get_termination
        from pymoo.termination.robust import RobustTermination
        from pymoo.termination.collection import TerminationCollection
        from pymoo.termination.ftol import SingleObjectiveSpaceTermination
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.core.mixed import (
            MixedVariableMating,
            MixedVariableDuplicateElimination,
            MixedVariableSampling,
        )
        from pymoo.optimize import minimize

        # TODO: We should probably infer the lower bounds for convergence using the model
        robust = RobustTermination(SingleObjectiveSpaceTermination(1e-8), period=20)
        termination = TerminationCollection(get_termination("time", "00:00:10"), robust)

        algorithm = GA(
            # Good heuristic: 10 times the number of variables, but at least 100
            pop_size=max(100, 10 * len(self.feature_values)),
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            # These values were found empirically
            crossover=SBX(prob=0.5, eta=15, vtype=float),
            mutation=PM(prob=0.1, eta=15, vtype=float),
        )

        problem = BayesianOptimizationProblem(
            feature_values=self.feature_values,
            feature_types=self.feature_types,
            input_features=self.input_features,
            model=model,
            partitioner=partitioner,
            acq=self.acq,
        )

        # Helper lambda to call pymoo and return a pd.DataFrame containing the generated solution
        def minimizer():
            out = minimize(problem, algorithm, termination)
            r = pd.Series(out.X, index=self.ordering).to_frame().T
            return r

        return problem, minimizer

    def _init_partitioner(self, samples: pd.DataFrame, model) -> tuple[KDTree, KDTreeSplitter]:

        assert samples is not None and len(samples) > 0

        input_values = {k: v for k, v in self.feature_values.items() if k in self.input_features}
        partitioner = KDTree(input_values, self.feature_types)

        objective = list(self.directions.keys())[0]
        # Split the initial tree a few time
        splitter = KDTreeSplitter(
            self.kernel, self.feature_values, self.feature_types, objective, self.directions, partitioner, model
        )
        splitter.variance_split(samples)

        for node in partitioner.nodes:
            if not node.is_leaf:
                continue
            best_in_node(self.directions[objective], samples, node, model)
        return partitioner, splitter

    def _iterate(self, samples: pd.DataFrame, nsamples: int):
        model = self._fit_model(samples)
        partitioner, splitter = self._init_partitioner(samples, model)

        problem, minimizer = self._build_minimizer(model, partitioner)
        niter = 0

        last_count = 0

        while len(samples) < nsamples:
            new_sample = minimizer()
            new_sample = self.kernel(new_sample)

            # Can occur if the sampling failed
            if new_sample is None or len(new_sample) == 0:
                continue

            samples = pd.concat([samples, new_sample])
            samples.reset_index(drop=True, inplace=True)

            # First, we refit the model with the new sample before
            # Taking the decision whether to split or not
            if niter % 10 == 0:
                model = self._fit_model(samples)
                problem.model = model
            else:
                objective = list(self.directions.keys())[0]
                model.update(new_sample.drop(columns=objective), new_sample[objective])

            if self.dump and (len(samples) - last_count) > self.dump:
                last_count = len(samples)
                self._dump(partitioner, samples)

            niter += 1

            samples, decision = splitter.cv_split(model, samples, new_sample)
            if self.output_dir is not None:
                samples.to_csv(self.output_dir / "samples.csv")

        return samples

    def __call__(self, samples: pd.DataFrame | None, nsamples: int):
        with tqdm(total=nsamples, leave=None, desc="Bayesian sampling") as pbar:
            if hasattr(self.kernel, "progress_bar"):
                self.kernel.progress_bar = pbar

            if samples is None:
                samples = self._boostrap(nsamples)

            samples = self._iterate(samples, nsamples)
        return samples
