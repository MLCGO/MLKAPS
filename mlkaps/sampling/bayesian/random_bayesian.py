"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableDuplicateElimination,
    MixedVariableSampling,
)
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination import get_termination
from mlkaps.modeling import encode_dataframe
from mlkaps.modeling import IQRVarianceEstimator
from mlkaps.sampling.generic_bounded_sampler import RandomSampler
import copy
from typing import Callable, Optional
from tqdm import tqdm
import logging
import time
from mlkaps.sampling import SamplerError
from mlkaps.sample_collection.samples_checkpoint import SamplesCheckpoint


class BayesianOptimizationProblem(Problem):

    def __init__(
        self,
        feature_values: dict[str],
        feature_types: dict[str, str],
        input_features: list[str],
        model,
        maximize: bool = False,
        **kwargs,
    ):
        self.model = model
        self.feature_values = feature_values
        self.feature_types = feature_types
        self.input_features = input_features
        self.ordering = sorted(list(feature_values.keys()))
        self._input_point = None
        self.maximize = maximize
        self.slambda = kwargs.pop("slambda", 2.5)

        mixed_vars = self._define_vars()
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
        design_parameters = [p for p in self.feature_values if p not in self.input_features]
        feature_values = {k: self.feature_values[k] for k in design_parameters}
        feature_types = {k: self.feature_types[k] for k in design_parameters}

        mixed_vars = {}
        for name, parameter_type in feature_types.items():
            match parameter_type:
                case "float":
                    pymoo_var = Real(bounds=feature_values[name].get_sampling_bounds())
                case "int":
                    pymoo_var = Integer(bounds=feature_values[name].get_sampling_bounds())
                case "Boolean":
                    pymoo_var = Binary()
                case "Categorical":
                    pymoo_var = Choice(options=feature_values[name].get_sampling_bounds())
                case _:
                    raise ValueError(f"Unexpected variable type for '{name}' ('{parameter_type}')")
            mixed_vars[name] = pymoo_var
        return mixed_vars

    @property
    def input_point(self):
        return self._input_point

    @input_point.setter
    def input_point(self, value):
        if isinstance(value, pd.DataFrame) and len(value) == 1:
            value = value.iloc[0]
        elif isinstance(value, (pd.Series, dict)):
            value = pd.Series(value)
        else:
            raise ValueError(f"Input points should be a pd.Series, dict or pd.DataFrame with one row, got {type(value)}")

        if any(col not in self.input_features for col in value.index):
            raise ValueError(
                f"Input points should only contain input features, got {value.index} with input features {self.input_features}"
            )

        self._input_point = value

    def _evaluate(self, x, out, *args, **kwargs):
        # First, for each X, finds the corresponding partition and the current best value
        # As we are in mixed variable mode, we receive an np.array of python dicts
        # We need to cast this to a list of dicts or pandas will be confused when building the DataFrame
        self.model.mean_only = False  # We want to predict the variance as well
        x = pd.DataFrame(list(x))

        for col in self.input_point.index:
            x[col] = self.input_point[col]

        mean, variance = self.model.predict(x)
        sigma = np.sqrt(variance)

        # We optimize the UCB (Upper Confidence Bound)
        # Computed as $$mu + lambda * sigma$$ when maximizing
        # and $$mu - lambda * sigma$$ when minimizing
        if self.maximize:
            ucb = -(mean + self.slambda * sigma)
        else:
            ucb = mean - self.slambda * sigma

        out["F"] = ucb


class RandomBayesianSampler:
    def __init__(
        self,
        kernel: Callable[[pd.DataFrame], pd.DataFrame],
        input_features: list[str],
        feature_values: dict[str],
        feature_types: dict[str, str],
        directions: dict[str, str],
        samples_checkpoint: SamplesCheckpoint,
        bootstrap_ratio=0.1,
        samples_per_iteration: int = 50,
        **kwargs,
    ):
        if not feature_values:
            raise SamplerError("Cannot build sampler with no features to sample")

        if not feature_types:
            raise ValueError("Cannot build sampler with no feature types")

        missing_inputs = [f for f in input_features if f not in feature_values]
        if missing_inputs:
            raise ValueError(f"Input features missing in feature values: {missing_inputs}")

        missing_types = [f for f in feature_values if f not in feature_types]
        if missing_types:
            raise ValueError(f"Feature types missing for features: {missing_types}")

        self.input_features = copy.deepcopy(input_features)
        self.design_parameters = [p for p in feature_values if p not in input_features]
        self.feature_values = copy.deepcopy(feature_values)
        self.feature_types = copy.deepcopy(feature_types)
        self.ordering = sorted(feature_values)

        self.kernel = kernel

        if len(directions) != 1:
            raise ValueError("RandomBayesianSampler only supports single-objective optimization for now")

        unsupported_directions = [v for v in directions.values() if v not in {"maximize", "minimize"}]
        if unsupported_directions:
            raise ValueError(f"Unsupported optimization directions: {unsupported_directions}")
        self.directions = copy.deepcopy(directions)

        if not (0 < bootstrap_ratio < 1.0):
            raise ValueError(f"bootstrap_ratio must be in (0, 1), got {bootstrap_ratio}")
        self.bootstrap_ratio = bootstrap_ratio

        self.samples_checkpoint = samples_checkpoint

        # FIXME: We only support one objective for now
        direction = list(directions.values())[0]
        # - When maximizing, we are interested in the upper bound of the confidence interval
        # - When minimizing, we are interested in the lower bound of the confidence interval
        self.hb_alpha = kwargs.pop("hb_alpha", 0.975 if direction == "maximize" else 0.025)

        self.do_early_stopping = kwargs.pop("do_early_stopping", True)

        self.samples_per_iteration = samples_per_iteration
        if self.samples_per_iteration <= 0:
            raise ValueError(f"Samples per iteration should be > 0, received {self.samples_per_iteration}")

        if len(kwargs) != 0:
            raise ValueError(f"Unknown parameters: {kwargs}")

    def _boostrap(self, nsamples: int):
        from mlkaps.sampling.generic_bounded_sampler import LhsSampler

        assert nsamples > 0, "Number of samples cannot be negative"

        sampler = LhsSampler(self.feature_types, self.feature_values)

        samples = sampler(int(nsamples * self.bootstrap_ratio))
        samples = self.kernel(samples)

        return samples

    def _fit_model(self, samples: pd.DataFrame):
        assert samples is not None and len(samples) > 0, f"Cannot fit model on None or empty dataframe {samples}"

        samples = encode_dataframe(self.feature_types, samples)
        features = list(self.feature_values.keys())

        # TODO: Expose the hb_alpha parameter in the constructor
        model = IQRVarianceEstimator(alpha=self.hb_alpha, method="forced_symmetry")
        # Fixme: We currently only support a single objective
        objective = list(self.directions.keys())[0]
        model.fit(samples[features], samples[objective])

        return model

    def _iterate(self, nsamples, samples, pbar) -> pd.DataFrame:
        pbar.set_description("IQR Bayesian Sampler")

        while len(samples) < nsamples:
            # Ensure we don't overshoot the total number of samples
            leftover_samples = min(self.samples_per_iteration, nsamples - len(samples))

            new_points = self._pick_bayesian_points(samples, leftover_samples)

            bayesian_points = self.kernel(new_points)

            samples = pd.concat(
                [
                    samples,
                    bayesian_points,
                ]
            )
            samples.reset_index(drop=True, inplace=True)
            # Ensure the current samples are consistent with the ones saved on disk
            # Note: the sampling backend has a reference to the checkpoint, and saving is done automatically
            # during sampling
            self.samples_checkpoint.consistency_check(samples)

        return samples

    def _pick_random_optimization_points(self, n_points: int) -> pd.DataFrame:
        """
        :param n_points: the number of optimization points to select
        :type n_points: int

        :return: a list of optimization points
        :rtype: pandas.DataFrame
        """

        sampler = RandomSampler(self.feature_types, self.feature_values, self.input_features)
        return sampler.sample(n_points)

    def _pick_bayesian_points(self, samples: pd.DataFrame, n_samples: int) -> Optional[pd.DataFrame]:

        if n_samples == 0:
            return None

        # First, pick new optimization points
        optimization_points = self._pick_random_optimization_points(n_samples)

        # Fit models to the currently sampled points
        model = self._fit_model(samples)

        maximize = list(self.directions.values())[0] == "maximize"

        slambda = 2.5  # Default value for the exploration-exploitation trade-off parameter

        problem = BayesianOptimizationProblem(
            self.feature_values,
            self.feature_types,
            self.input_features,
            model,
            maximize=maximize,
            slambda=slambda,
        )
        algorithm = NSGA2(
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        if self.do_early_stopping:
            model.mean_only = True  # We only need the mean for early stopping
            termination = self._build_early_stopping_criterion(model)
            termination = TerminationCollection(termination, get_termination("time", "0:0:10"))
        else:
            termination = get_termination("time", "0:0:10")

        # Run the optimizer on each of the optimization points
        sampling_list = []
        logging.info(f"Running Bayesian optimization on {len(optimization_points)} points")
        for _, point in optimization_points.iterrows():
            local_optimum, _ = self._run_bayesian_on_point(point, algorithm, problem, termination=termination)

            # Append the local solution to the list of points to be sampled
            sampling_list.append(local_optimum)

        # Aggregate all the results in a DataFrame
        sampling_list = pd.DataFrame(sampling_list)
        sampling_list = pd.concat([optimization_points, sampling_list], axis=1)

        return sampling_list

    def _build_early_stopping_criterion(self, surogate_models) -> RobustTermination:
        """Build a stopping criterion with an heuristic for the convergence threshold

        Execute 10k random solutions, and take a fraction of the minimum value as a threshold

        :param genetic_config: The configuration of the optimizer
        :type genetic_config: GeneticOptimizerConfig
        :param surogate_models: The models to compute the threshold with
        :type surogate_models: dict
        :return: A convergence stopping criterion
        :rtype: RobustTermination
        """

        if not isinstance(surogate_models, dict):
            surogate_models = {"default": surogate_models}

        begin = time.time()

        sampler = RandomSampler(self.feature_types, self.feature_values)

        samples = sampler.sample(1000000)

        predictions = None
        for m in surogate_models.values():
            pred = m.predict(samples)
            if predictions is None:
                predictions = pred
            else:
                np.column_stack([pred, predictions])

        # Take the nearest power of 10 below the minimum prediction in absolute value
        epsilon = 1e-10  # Small value to avoid log10(0)
        thresh = 10 ** (np.floor(np.log10(np.min(abs(predictions)) + epsilon)) - 1)

        end = time.time()

        logging.info(f"Early stopping enabled, threshold inferred to be {thresh} (Overhead: {np.round(end - begin, 3)}s)")
        return RobustTermination(MultiObjectiveSpaceTermination(tol=thresh, n_skip=5), period=20)

    def _run_bayesian_on_point(self, point, algorithm, problem, termination):
        """
        Execute the given genetic algorithm on one optimization point

        :param point: The optimization point to run the GA on
        :type: pandas.Series
        :param algorithm: The genetic algorithm to execute
        :type algorithm: pymoo.algorithms.moo.nsga2.NSGA2
        :param problem: The pymoo problem corresponding to the optimization job
        :type problem: mlkaps.optimization.genetic_optimizer.DesignParametersProblem

        :return: The optimal configuration found for the given point
        :rtype: dict
        """

        problem.input_point = point

        # Run the minimization task with a short timeout to add some uncertainty
        local_optimum = minimize(problem, algorithm, termination=termination)
        if isinstance(local_optimum.X, dict):
            local_optimum = local_optimum.X
        else:
            idx = np.argmin(local_optimum.F)
            local_optimum = local_optimum.X[idx]
        return local_optimum, point

    def __call__(self, samples: pd.DataFrame | None, nsamples: int):
        with tqdm(total=nsamples, leave=None, desc="Bayesian sampling") as pbar:
            if hasattr(self.kernel, "progress_bar"):
                self.kernel.progress_bar = pbar
            samples = self.samples_checkpoint.maybe_load_samples()
            n_bootstrap = int(nsamples * self.bootstrap_ratio)

            if samples is not None:
                n_bootstrap = max(0, n_bootstrap - len(samples))

            if n_bootstrap > 0:
                samples = self._boostrap(n_bootstrap)

            samples = self._iterate(nsamples, samples, pbar)
        return samples
