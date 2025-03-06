"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
import pandas as pd
import pymoo.termination
from tqdm import tqdm
import os
import logging
import time

import pymoo.core.result
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableGA,
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Real, Binary, Integer
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination

from mlkaps.configuration import ExperimentConfig
from mlkaps.modeling.encoding import encode_dataframe
from mlkaps.sampling.sampler_factory import SamplerFactory

from mlkaps.sampling import RandomSampler


class GeneticOptimizerConfig:
    """
    Configuration object for the genetic optimizer module

    This class is used to store all parameters related to the genetic optimizer, as well as
    the parsing logic from the configuration dictionary.
    """

    def __init__(
        self,
        experiment_configuration: ExperimentConfig,
        sampler,
        samples_count,
        do_early_stopping=False,
    ):
        """
        Create a new GeneticOptimizerConfig object

        Parameters
        ----------
        experiment_configuration: ExperimentConfig
            The global configuration of the experiment

        sampler: Sampler
            The sampler to use to generate the optimization points

        samples_count: int
            The number of optimization points to generate

        """

        self.experiment_configuration = experiment_configuration

        # Default every objective to the same weight
        self.normalization_coefficients = {
            k: 1 for k in experiment_configuration.objectives
        }

        # Default parameters
        # FIXME: This should be a parameter
        self.termination_criterion = get_termination("time", "00:00:30")
        self.optimization_parameters = {}

        # FIXME: This should be a parameter
        self.selection_method = "normalized"
        self.sampler = sampler
        sampler.set_variables(
            experiment_configuration.parameters_type,
            experiment_configuration["parameters"]["features_values"],
            mask=experiment_configuration.input_parameters,
        )
        self.samples_count = samples_count

        self.do_early_stopping = do_early_stopping

    @staticmethod
    def _parse_genetic_optimization(config, config_dict: dict):
        """
        Extract the genetic optimization parameters from the configuration dictionary, and update
        the configuration object accordingly
        """
        if "optimization_parameters" not in config_dict:
            raise Exception(
                "Missing optimization_parameters section in the configuration dict"
            )

        parameter_section = config_dict["optimization_parameters"]

        config.selection_method = parameter_section.get(
            "selection_method", "normalized_selection"
        )

        # Load algorithm specific parameters
        config.optimization_parameters = parameter_section.get("evolution", {})

        # Build the termination criterion as a pymoo collection of termination object
        termination = parameter_section.get("termination", {})
        terminations = [get_termination(k, v) for k, v in termination.items()]

        config.termination_criterion = TerminationCollection(*terminations)
        if "selection_parameters" not in parameter_section:
            return

        selection_parameters = parameter_section["selection_parameters"]
        coefficients = selection_parameters.get("coefficients", {})

        # Handle feature-specific selection coefficient
        for obj in coefficients:
            if obj not in config.experiment_configuration.objectives:
                raise Exception(f'Coefficient set for undefined objective "{obj}"')
            config.normalization_coefficients[obj] = selection_parameters[
                "coefficients"
            ][obj]

    @staticmethod
    def from_configuration_dict(config_dict: dict, exp_config: ExperimentConfig):
        """
        Parse a GeneticOptimizerConfig object from a configuration dictionary
        """

        optim_section = config_dict["OPTIMIZATION"]
        optimization_method = optim_section["optimization_method"]

        # Ensure that the configuration is for a genetic optimizer
        if optimization_method != "genetic":
            raise Exception(
                "Tried to parse a genetic optimizer configuration, but the configuration dict "
                f"specifies a '{optimization_method}' optimization method"
            )

        sampler = SamplerFactory(exp_config).from_config(
            optim_section["sampling"]["sampler"]["sampling_method"]
        )
        sampler.set_variables(
            exp_config.parameters_type,
            exp_config["parameters"]["features_values"],
            mask=exp_config.input_parameters,
        )
        sample_count = optim_section["sampling"]["sample_count"]
        early_stopping = optim_section["optimization_parameters"].get(
            "early_stopping", False
        )

        # Build the configuration object with basic parameters
        res = GeneticOptimizerConfig(
            exp_config, sampler, sample_count, do_early_stopping=early_stopping
        )
        # Parse additional parameters
        GeneticOptimizerConfig._parse_genetic_optimization(res, optim_section)

        return res


class DesignParametersProblem(Problem):
    """
    Custom pymoo problem that uses the models generated in the modeling phase to estimate
    the objective values for a set of design parameters. Those objectives are used to evaluate a
    population of models to find the best parameters for a given kernel inputs
    """

    def __init__(self, configuration: ExperimentConfig, objectives_models: dict):
        """
        Initialize the problem with the configuration and the models

        Parameters
        ----------
        configuration:
            The global configuration of the experiment

        objectives_models:
            The models generate in the modeling phase
        """

        # Kernels inputs will be defined later on
        self.kernel_inputs = None
        self.input_columns = None

        self.configuration = configuration
        # Extract the name and type of the optimization feature
        # FIXME: this probably should be a method in the configuration
        self.optimization_parameters = {
            k: t
            for k, t in configuration.parameters_type.items()
            if k in configuration.design_parameters
        }

        self.model_type = configuration["modeling"]["modeling_method"]
        self.surrogate_models = objectives_models
        self.objectives_count = len(self.configuration.objectives)

        if self.objectives_count > 2:
            raise ValueError(
                "Only 1D and 2D optimization problems are currently supported"
            )

        self.objectives_directions = configuration["experiment"][
            "objectives_directions"
        ]

        mixed_vars = self._define_vars()
        super().__init__(vars=mixed_vars, n_obj=self.objectives_count)

    def _define_vars(self):
        # To define a mixed precision problem, we need to define each
        # variable, and their respective bound
        feature_values = self.configuration["parameters"]["features_values"]

        mixed_vars = {}
        for name, parameter_type in self.optimization_parameters.items():
            match parameter_type:
                case "float":
                    pymoo_var = Real(bounds=feature_values[name])
                case "int":
                    pymoo_var = Integer(bounds=feature_values[name])
                case "Boolean":
                    pymoo_var = Binary()
                case "Categorical":
                    pymoo_var = Choice(options=feature_values[name])
                case _:
                    raise ValueError(
                        f"Unexpected variable type for '{name}' ('{parameter_type}')"
                    )
            mixed_vars[name] = pymoo_var
        return mixed_vars

    def set_kernel_input(self, kernel_inputs):
        """
        Set the input coordinate for the problem, to which the design parameters will be applied
        """
        self.kernel_inputs = kernel_inputs
        if isinstance(self.kernel_inputs, pd.DataFrame):
            self.input_columns = kernel_inputs.columns
        elif isinstance(self.kernel_inputs, pd.Series):
            self.input_columns = self.kernel_inputs.index
        elif isinstance(self.kernel_inputs, dict):
            # If the input is dict, we can directly build a Dataframe from it
            self.input_columns = None

    def _build_model_input(self, x):
        # Pymoo returns a ndarray of dict
        # We must convert it to a simple list for pandas to automatically builds the DataFrame
        x = list(x)
        sample_count = len(x)
        # Create a dataframe containing the samples
        model_input = pd.DataFrame(x)

        # Tile the user inputs to match the number of samples
        repeats_user_inputs = pd.DataFrame(
            np.tile(self.kernel_inputs, (sample_count, 1)), columns=self.input_columns
        )

        # Concat both dataframes
        model_input = pd.concat([model_input, repeats_user_inputs], axis=1)

        return model_input

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objectives of the given population

        Parameters

        x:
            a 2D array of shape (n_features, n_samples)
        out:
            The evaluation of each objective for every sample
        args:
            Extra arguments, unused in this implementation
        kwargs:
            Extra keyword arguments, unused in this implementation
        """

        # Ensure the user has set the kernel inputs
        if self.kernel_inputs is None:
            raise Exception("Kernel inputs were not set prior to calling evaluate !")

        model_inputs = self._build_model_input(x)

        predictions = []

        # Build all the predictions
        for i in self.configuration.objectives:
            prediction = self.surrogate_models[i].predict(model_inputs)

            # If one of the objective is a mazimization objective, then reverse it
            if self.objectives_directions[i] == "maximize":
                prediction *= -1
            predictions.append(prediction)

        if len(self.configuration.objectives) == 1:
            out["F"] = predictions[0]
        elif len(self.configuration.objectives) == 2:
            out["F"] = np.column_stack(predictions)
        else:
            raise Exception(
                f"Unsupported number of objectives ({len(self.configuration.objectives)})"
            )


class _GeneticOptimizationMethod:
    """
    Base class for all genetic optimization methods.
    Optimization methods apply an optimization algorithm for one input point, and differ
    in the algorithm used/ logic for selecting the final solutions
    """

    def run(self, kernel_input: pd.Series):
        raise NotImplementedError()


class _NormalizedOptimizationMethod(_GeneticOptimizationMethod):
    """
    Optimization method that uses a normalized weighted sum of the objectives to select the
    best solution
    """

    def __init__(self, genetic_config: GeneticOptimizerConfig, surrogate_models):
        self.config = genetic_config
        self.surrogate_models = surrogate_models
        self.exp_config = self.config.experiment_configuration

    def _normalized_selection(self, raw_parameters, raw_objectives):
        """
        Parse a list of results and select the one with the best normalized weighted sum,
         where the weights correspond to user-defined coefficients for each objective

        Parameters
        ----------
        raw_parameters:
            A list of encoded kernel design parameters

        raw_objectives:
            A list of corresponding objective values
            The optimal design parameters and corresponding objective values

        Returns
        -------
        tuple:
            optimal_parameters
                The optimal design parameters

            optimal_values
                The corresponding objective values
        """

        # FIXME: We should generalize this to any number of objectives
        if len(self.config.experiment_configuration.objectives) != 2:
            raise Exception("Normalized selection requires exactly 2 objectives !")

        # Normalize the objectives, and find the best parameter set

        normalized_objectives = raw_objectives.copy()
        # Normalize both objectives
        normalized_objectives[:, 0] = np.interp(
            raw_objectives[:, 0],
            (raw_objectives[:, 0].min(), raw_objectives[:, 0].max()),
            (0, 1),
        )

        normalized_objectives[:, 1] = np.interp(
            raw_objectives[:, 1],
            (raw_objectives[:, 1].min(), raw_objectives[:, 1].max()),
            (0, 1),
        )

        coefficients = self.config.normalization_coefficients
        first_coefficient = list(coefficients.values())[0]
        second_coefficient = list(coefficients.values())[1]

        # Find the parameter set with the best normalized weighted sum
        best_normalized_index = np.argmin(
            first_coefficient * normalized_objectives[:, 0]
            + second_coefficient * normalized_objectives[:, 1]
        )

        optimal_parameters = pd.Series(
            raw_parameters[best_normalized_index],
            index=self.exp_config.design_parameters,
        )

        # We return the raw objectives, not the normalized ones
        optimal_values = raw_objectives[best_normalized_index]

        return optimal_parameters, optimal_values

    def _run_nsga2(self, problem) -> pymoo.core.result.Result:
        """
        Run the NSGA2 algorithm on the given problem,
        extracting required parameters from the configuration

        Parameters
        ----------

        problem:
            The problem to run the algorithm on

        Returns
        --------

        pymoo.core.result.Result:
            The return value of the NSGA2 algorithm
        """

        # Create the algorithm object
        # NSGA2 For mixed variables
        algorithm = NSGA2(
            **self.config.optimization_parameters,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        # Run the algorithm
        res = minimize(
            problem, algorithm, termination=self.config.termination_criterion, seed=1
        )
        return res

    def run(self, kernel_input):
        """
        Runs the NSGA2 algorithm and selects the kernel design parameters using
        the user-defined coefficients for each objective

        Parameters
        ----------
        kernel_input:
            The kernel inputs corresponding to the local sampling point

        Returns
        -------
        tuple:
            optimal_parameters:
                The optimal design parameter

            optimal_objectives_values
                the corresponding objective values
        """
        # FIXME: Normalized optimization can support as many as objectives as the user wants
        if len(self.config.experiment_configuration.objectives) != 2:
            raise Exception(
                "Normalized optimization currently only supports 2D optimization problems"
            )

        problem = DesignParametersProblem(
            self.config.experiment_configuration, self.surrogate_models
        )
        problem.set_kernel_input(kernel_input)

        res = self._run_nsga2(problem)
        optimal_parameters, optimal_objectives_values = self._normalized_selection(
            res.X, res.F
        )

        return optimal_parameters, optimal_objectives_values


class _MonoObjectiveOptimizationMethod(_GeneticOptimizationMethod):
    """
    An optimization method for mono-objective problem, where the best solution is selected
    based on the minimum objective value
    """

    def __init__(
        self,
        genetic_config: GeneticOptimizerConfig,
        surogate_models,
        record_history=True,
    ):
        self.config = genetic_config
        self.surrogate_models = surogate_models
        self.do_record = record_history

        if genetic_config.do_early_stopping:
            criterion = self._build_early_stopping_criterion(
                genetic_config, surogate_models
            )
            self.termination = TerminationCollection(
                genetic_config.termination_criterion, criterion
            )
        else:
            self.termination = genetic_config.termination_criterion

    def _build_early_stopping_criterion(
        self, genetic_config: GeneticOptimizerConfig, surogate_models
    ) -> RobustTermination:
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

        exp_config = genetic_config.experiment_configuration
        sampler = RandomSampler(exp_config.parameters_type, exp_config.feature_values)

        samples = sampler.sample(10000)

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

        logging.info(
            f"Early stopping enabled, threshold inferred to be {thresh} (Overhead: {np.round(end - begin, 3)}s)"
        )
        return RobustTermination(
            MultiObjectiveSpaceTermination(tol=thresh, n_skip=5), period=20
        )

    def run(self, kernel_input: pd.Series):
        """
        Optimized kernel optimization for single objective experiments

        Parameters
        ----------
        kernel_input:
            The kernel inputs corresponding to the local sampling point

        Returns
        -------
        tuple:
            X:
                The optimal design parameters
            F:
                The corresponding objective values
        """

        if len(self.config.experiment_configuration.objectives) != 1:
            raise Exception(
                "Mono objective optimization was used with multiples objectives"
            )

        problem = DesignParametersProblem(
            self.config.experiment_configuration, self.surrogate_models
        )
        problem.set_kernel_input(kernel_input)

        algorithm = MixedVariableGA(**self.config.optimization_parameters)

        res = minimize(
            problem,
            algorithm,
            termination=self.termination,
            seed=1,
            save_history=self.do_record,
        )

        if self.do_record:
            self._record_history(res, kernel_input)

        best_configuration = pd.concat([pd.Series(res.X), kernel_input])

        optimal_index = np.argmin(res.F)
        return best_configuration, res.F[optimal_index]

    def _record_history(self, ga_res, kernel_input):
        history = ga_res.history

        dbs = []

        input_df = pd.DataFrame([kernel_input], columns=kernel_input.index).reset_index(
            drop=True
        )

        # Record the best solution at each iteration
        for iteration, hist in enumerate(history):
            best_sol_index = np.argmin(hist.pop.get("F"))

            # Build a DataFrame containing the best solution
            db = pd.DataFrame([hist.pop.get("X")[best_sol_index]])
            db["performance"] = hist.pop.get("F")[best_sol_index]
            db["iteration"] = iteration

            dbs.append(pd.concat((db, input_df), axis=1))

        db = pd.concat(dbs, axis=0).reset_index(drop=True)

        # Check for existing records to append to
        output_path = (
            self.config.experiment_configuration.output_directory
            / "ga_convergence_study/records.csv"
        )
        if output_path.exists():
            db_old = pd.read_csv(output_path)
            db = pd.concat([db_old, db], axis=0).reset_index(drop=True)
        else:
            os.makedirs(output_path.parent, exist_ok=True)

        db.to_csv(output_path, index=False)


class GeneticOptimizer(object):
    """
    A genetic optimizer, that create a list of samples inside the sampling space and
    find the local optimal design parameters, gathered as a dataframe.
    """

    def __init__(self, configuration: GeneticOptimizerConfig, surrogate_models: dict):
        """
        Construct a new genetic optimizer based on the passed global
        configuration

        Parameters
        ----------
        configuration:
            The global configuration of the experiment

        surrogate_models:
            A dict of surrogates for each objective in the experiment, defined as
            {objective_name: surrogate_model}
        """

        self.config = configuration
        self.exp_configuration = configuration.experiment_configuration

        self.surrogate_models = surrogate_models

        self.sampler = configuration.sampler

    def _make_optimization_method(self):
        """
        Finds the optimization method to use for the current configuration

        Returns
        -------
        functor:
            A functor to an optimization method
        """

        selection_method = self.config.selection_method
        objectives = self.config.experiment_configuration.objectives
        # If we only have one objective, we can use the mono-objective optimization method
        if len(objectives) == 1 or selection_method == "mono":
            return _MonoObjectiveOptimizationMethod(self.config, self.surrogate_models)
        elif self.config.selection_method == "normalized":
            return _NormalizedOptimizationMethod(self.config, self.surrogate_models)
        else:
            raise ValueError(f"Unknown selection method ('{selection_method}')")

    def _optimize_point(
        self,
        optimization_method: _GeneticOptimizationMethod,
        input_features: pd.Series,
    ):
        best_design_params, objective_values = optimization_method.run(input_features)
        return best_design_params, objective_values

    def _optimize_all_samples(
        self, optimization_method, optimization_points: pd.DataFrame
    ):
        """
        Iterate over all the given samples, and find the best design parameters for each sample.
        Results are returned as a DataFrame

        Parameters
        ----------
        optimization_method:
            The function to use to find the best design parameters for each
            sample.

        optimization_points: pd.DataFrame
            A dataframe containing the samples to optimize, in the shape
            (nb_features, nb_samples).
        progress_bar:
            a progress bar to display the progress.

        Returns
        -------
        pd.DataFrame:
            A dataframe containing the results of the optimization, in the shape (nb_features,
            nb_samples).
        """

        results = []

        # Run the optimization method on every points
        for _, user_inputs in tqdm(
            optimization_points.iterrows(),
            total=len(optimization_points),
            desc="Running optimization phase",
        ):
            best_config, _ = self._optimize_point(optimization_method, user_inputs)
            results.append(best_config)

        results = pd.DataFrame(results)
        return results

    def _optimize(self):

        # First define the optimization points
        samples = self.sampler.sample(self.config.samples_count)

        # Build the optimization method
        optimization_method = self._make_optimization_method()

        # Run the optimization method on every points
        results = self._optimize_all_samples(optimization_method, samples)

        return encode_dataframe(self.exp_configuration.parameters_type, results)

    def _save_results(self, results):
        output_path = self.exp_configuration.output_directory / "optim.csv"
        results.to_csv(output_path, index=False)

    def run(self):
        """
        Generate a grid of samples for the kernel inputs, and find the best parameters
        of every sample. Further down, those parameters can be clustered to form a map of the best
        design parameters for each kernel input.

        Returns
        -------
        pd.DataFrame:
            A dataframe with the best design parameters for each kernel input.
        """

        results = self._optimize()
        self._save_results(results)

        return results
