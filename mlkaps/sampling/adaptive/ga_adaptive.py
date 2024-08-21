"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

"""
Definition of the GA-Adaptive sampling process, based on genetic algorithms
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableDuplicateElimination,
    MixedVariableSampling,
)
from pymoo.optimize import minimize
from tqdm import tqdm
from typing import Callable


from mlkaps.configuration import ExperimentConfig
from mlkaps.modeling import OptunaTunerLightgbm, SurrogateFactory
from mlkaps.optimization.genetic_optimizer import DesignParametersProblem
from . import HVSampler
from .. import SamplerError
from .. import RandomSampler, LhsSampler
import logging


import pymoo.termination
from tqdm import tqdm
import os
import logging
import time
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination import get_termination


class GAAdaptiveSampler:
    """
    Adaptive sampler based on genetic algorithms.

    The sampler uses a genetic algorithm to pick interesting points in the design space,
    and combines this with a HVS sampler to explore the design space.
    """

    def __init__(
        self,
        execution_function: Callable[[pd.DataFrame], pd.DataFrame],
        configuration: ExperimentConfig,
        n_samples: int,
        samples_per_iteration: int,
        bootstrap_ratio: float,
        initial_ga_ratio: float,
        final_ga_ratio: float,
        do_early_stopping: bool = True,
        use_optuna: bool = False,
    ):
        """
        Initialize the GA-Adaptive sampler

        :param execution_function: A callback for the execution samples that will evaluates the samples
        :type execution_function: Callable[[pandas.DataFrame], pandas.DataFrame]
        :param configuration: The experiment configuration object associated with the experiment
        :type configuration: mlkaps.configuration.ExperimentConfig
        :param n_samples: The total number of samples to take
        :type n_samples: int
        :param samples_per_iteration: The number of samples to take per iterations of the GA
            loop
        :type samples_per_iteration: int
        :param bootstrap_ratio: The ratio (value between 0-1) of the total number of samples to take
            in the bootstraping phase
        :type bootstrap_ratio: float
        :param initial_ga_ratio: The ratio (value between 0-1)  of points taken with the GA Algorithm
            at the first iteration of the algorithm. The ratio at iteration x is the linear interpolation
            between this value and final_ga_ratio
        :type initial_ga_ratio: float
        :param final_ga_ratio: The ratio of points taken with the GA Algorithm
            at the last iteration of the algorithm. See initial_ga_ratio.
        """

        self.config = configuration
        self.features = configuration["parameters"]["features_values"]
        self.input_features = configuration.input_parameters
        self.execution_function = execution_function

        self.n_samples = n_samples
        self.samples_per_iteration = int(samples_per_iteration)

        self.bootstrap_ratio = bootstrap_ratio
        self.initial_ga_ratio = initial_ga_ratio
        self.final_ga_ratio = final_ga_ratio

        self.do_early_stopping = do_early_stopping
        self.use_optuna = use_optuna
        self._verify_input()

        # HVS sampler for exploration
        self.hvs_sampler = HVSampler(
            self.config.parameters_type,
            self.config["parameters"]["features_values"],
            error_metric="cov",
        )

        # FIXME: dirty quick-restart
        self.output_path = self.config.output_directory / "kernel_sampling/samples.csv"

        self.models = {}
        self.iteration = 0

    def _verify_input(self):
        """
        Ensure that the parameters used to build this model are valid

        :raise SamplerError: If the sampler was built with incorrect parameters
        """

        if (
            self.samples_per_iteration > self.n_samples
            or self.samples_per_iteration < 1
        ):
            raise SamplerError("samples_per_iteration must be between 1 and n_samples")

        if self.bootstrap_ratio > 1 or self.bootstrap_ratio < 0:
            raise SamplerError("bootstrap_ratio must be between 0 and 1")

        if self.initial_ga_ratio > 1 or self.initial_ga_ratio < 0:
            raise SamplerError("initial_ga_ratio must be between 0 and 1")

        if self.final_ga_ratio > 1 or self.final_ga_ratio < 0:
            raise SamplerError("final_ga_ratio must be between 0 and 1")

        if self.final_ga_ratio < self.initial_ga_ratio:
            raise SamplerError("final_ga_ratio must be greater than initial_ga_ratio")

    def __call__(self) -> pd.DataFrame:
        """
        Run the sampling process using the parameters used in the constructor.
        First bootstrap using LHS, then run the main sampling loop using a combination
        of LHS and genetic algorithm.

        :raise SamplerError: if an error occurs during the sampling process
        :return: a list of samples and their respectives values
        :rtype: pandas.DataFrame
        """

        with tqdm(total=self.n_samples, leave=None) as pbar:
            # FIXME: The execution function should be updated to have a proper interface for such cases
            # Attempt to set the progress bar on the execution function
            if hasattr(self.execution_function, "progress_bar"):
                self.execution_function.progress_bar = pbar

            try:
                logging.info("GA-Adaptive started")

                n_bootstrap = int(self.n_samples * self.bootstrap_ratio)
                samples = self._maybe_load_samples()
                if samples is not None:
                    pbar.update(len(samples))
                    n_bootstrap = n_bootstrap - len(samples)

                logging.info("Bootstrapping with LHS")

                if n_bootstrap > 0:
                    lhs_samples = self._lhs_bootstrap(n_bootstrap, pbar)
                    samples = pd.concat([samples, lhs_samples])

                samples.to_csv(self.output_path, index=False)

                logging.info("Bootstrapping finished, starting GA-Adaptive loop")
                samples = self._resampling_loop(samples, pbar)
                return samples
            except Exception as exc:
                # Wrap any exception in a SamplerError
                raise SamplerError("GA-Adaptive sampling failed!") from exc

    def _maybe_load_samples(self):
        if not self.output_path.exists():
            return None

        logging.info(f"Found samples at '{self.output_path}', quick-restarting")
        logging.warning(
            f"GA-Adaptive will quick-restart by default, this is currently not configurable\nPlease delete '{self.output_path} to skip quick-restart."
        )
        loaded_samples = pd.read_csv(self.output_path)
        return loaded_samples

    def _lhs_bootstrap(self, n_samples, pbar):
        if n_samples <= 0:
            return None

        # Bootstrap the sampling with an LHS
        pbar.set_description("GA-Adaptive: bootstrapping with LHS")

        sampler = LhsSampler(self.config.parameters_type, self.features)
        lhs_samples = sampler.sample(n_samples)

        return self._sample_kernel(lhs_samples)

    def _resampling_loop(self, samples, pbar) -> pd.DataFrame:
        pbar.set_description("GA-Adaptive")

        final_ratio_delta = self.final_ga_ratio - self.initial_ga_ratio
        while len(samples) < self.n_samples:
            # Ensure we don't overshoot the total number of samples
            leftover_samples = min(
                self.samples_per_iteration, self.n_samples - len(samples)
            )

            # We want to start with a high ratio of HVS picked points, and gradually decrease it
            # in favor of the GA optimized points
            # However, we don't want the number of GA points to be 0 at the start, start with 20%
            curr_ratio = self.initial_ga_ratio + final_ratio_delta * (
                len(samples) / self.n_samples
            )

            n_ga_points = int(np.round(curr_ratio * leftover_samples))

            new_points = self._pick_ga_points(samples, n_ga_points)

            # We randomly pick the remaining points for the exploration component
            # Of the sampler
            if n_ga_points == 0:
                delta = leftover_samples
            else:
                delta = max(0, leftover_samples - len(new_points))
            hvs_samples = self._pick_hvs_samples(delta, samples)

            # Concat GA points with HVS picked samples
            # Note that the HVS sampler already concatenates the new points with the old ones,
            # so no need to do it here

            ga_points = self._sample_kernel(new_points)

            samples = pd.concat(
                [
                    hvs_samples,
                    ga_points,
                ]
            )

            # FIXME: dirty quick-restart
            samples.to_csv(self.output_path, index=False)

            self.iteration += 1

        return samples

    def _sample_kernel(self, new_points: pd.DataFrame) -> pd.DataFrame | None:
        """
        Execute the new_points using the execution function, returns None
        if no points was passed.


        :return: the list of samples decorated with their values
            One should not expect this function return to match the inputs, as samples
            may have failed and have been removed from the resulting DataFrame
        :rtype: pandas.DataFrame
        """

        if new_points is None or len(new_points) == 0:
            return None
        return self.execution_function(new_points)

    def _pick_random_optimization_points(self, n_points: int) -> pd.DataFrame:
        """
        Randomly pick new points to run the GA on

        :param n_points: the number of optimization points to select
        :type n_points: int

        :return: a list of optimization points
        :rtype: pandas.DataFrame
        """

        sampler = RandomSampler(
            self.config.parameters_type, self.features, self.input_features
        )
        return sampler.sample(n_points)

    def _pick_hvs_samples(self, n_samples: int, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the HVS subsampler to find new points for the next sampling iteration

        :param n_samples: The number of samples to take using HVS
        :type n_samples: int
        :param data: All the samples collected so far
        :type data: pandas.DataFrame

        :return: A list of samples taken with HVS
        :rtype: pandas.DataFrame
        """
        new_samples = self.hvs_sampler.sample(n_samples, data, self.execution_function)
        return new_samples

    def _pick_ga_points(self, samples: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Compute new samples using GA. First, we randomly select new optimization points.
        We then build new surrogate models based on the current samples.
        We then run NSGA2 on those points, using the surrogate models as oracles.
        The samples returned are the optimal points founds by the GA.

        :param samples: A list of currently sampled points
        :type samples: pandas.DataFrame
        :param n_samples: The number of samples to take using GA
        :type n_samples: int

        :return: A list of new samples for evaluation
        :rtype: pandas.DataFrame
        """

        if n_samples == 0:
            return None

        # First, pick new optimization points
        optimization_points = self._pick_random_optimization_points(n_samples)

        # Fit models to the currently sampled points
        models = self._fit_models(samples)

        # Create the GA object
        problem = DesignParametersProblem(self.config, models)
        algorithm = NSGA2(
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        if self.do_early_stopping:
            termination = self._build_early_stopping_criterion(self.models)
            termination = TerminationCollection(
                termination, get_termination("time", "0:0:10")
            )
        else:
            termination = get_termination("time", "0:0:10")

        # Run GA on every optimization points
        sampling_list = []
        for _, point in tqdm(
            optimization_points.iterrows(),
            total=len(optimization_points),
            desc="Running Genetic Algorithm",
            leave=None,
        ):
            local_optimum, _ = self._run_ga_on_point(
                point, algorithm, problem, termination=termination
            )

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

        begin = time.time()

        exp_config = self.config
        sampler = RandomSampler(exp_config.parameters_type, exp_config.feature_values)

        samples = sampler.sample(1000000)

        predictions = None
        for m in surogate_models.values():
            pred = m.predict(samples)
            if predictions is None:
                predictions = pred
            else:
                np.column_stack([pred, predictions])

        # Take the nearest power of 10 below the minimum prediction in absolute value
        thresh = 10 ** (np.floor(np.log10(np.min(abs(predictions)))) - 1)

        end = time.time()

        logging.info(
            f"Early stopping enabled, threshold inferred to be {thresh} (Overhead: {np.round(end - begin, 3)}s)"
        )
        return RobustTermination(
            MultiObjectiveSpaceTermination(tol=thresh, n_skip=5), period=20
        )

    def _run_ga_on_point(self, point, algorithm, problem, termination):
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

        problem.set_kernel_input(point)

        # Run the minimization task with a short timeout to add some uncertainty
        local_optimum = minimize(problem, algorithm, termination=termination)
        if isinstance(local_optimum.X, dict):
            local_optimum = local_optimum.X
        else:
            # Randomly pick one of the solutions if there are multiple, so we avoid
            # having the same point multiple times, and it allows us to discover new
            # potential optimums
            local_optimum = local_optimum.X[np.random.choice(local_optimum.X.shape[0])]
        return local_optimum, point

    def _build_model(self, obj, samples):

        if self.use_optuna:
            tuner = OptunaTunerLightgbm(
                samples.drop(self.config.objectives, axis=1), samples[obj]
            )
            model, _ = tuner.run(time_budget=2 * 60, n_trials=128)
        else:
            factory = SurrogateFactory(self.config, samples)
            model = factory.build(obj)
        return model

    def _fit_models(self, samples):
        """
        Create new LightGBM models and fit them to the current samples

        :param samples: The list of samples to train the new models on
        :type samples: pandas.DataFrame

        :return: A dictionnary containing one model per objective
        :rtype: dict
        """

        for obj in self.config.objectives:
            first_iter = obj not in self.models

            if first_iter or (self.iteration % 4) == 0:
                self.models[obj] = self._build_model(obj, samples)
            else:
                self.models[obj].fit(
                    samples.drop(self.config.objectives, axis=1), samples[obj]
                )

        return self.models
