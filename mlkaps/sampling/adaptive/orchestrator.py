"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import time
from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from .adaptive_sampler import AdaptiveSampler


class StoppingCriterion:
    """
    Base class for stopping criterion used by the AdaptiveSampler Orchestrator

    A stopping criterion is a function that takes as input the current data and the current
    error, and returns
    true if the sampling should stop, false otherwise. It can also limit the number of samples
    taken on a given
    iteration depending on the criterion.
    """

    def init(self) -> None:
        """
        Resets and initialize the state of the criterion, Called before the start of the sampling
        process
        """
        raise NotImplementedError()

    def is_reached(self, data: pd.DataFrame, current_error: pd.DataFrame) -> bool:
        """
        Check if the stopping criterion is reached

        Parameters
        ----------
        data
            A dataframe containing all the sampled points so far
        current_error
            The error of the current points, stored as a dataframe with one column per

        Returns
        -------
        True if the stopping criterion is reached, False otherwise

        """
        raise NotImplementedError()

    def max_samples(self, data: pd.DataFrame | None) -> int:
        """
        Returns the maximum number of samples that can be taken on a given iteration
        Parameters
        ----------
        data A dataframe containing all the sampled points so far

        Returns
        -------
        -1 if there is no limit, or the maximum number of samples that can be taken on a given
        iteration otherwise

        """
        raise NotImplementedError()

    def get_progression(self, data: pd.DataFrame | None) -> float:
        """
        Compute the current percentage progression relative to this criterion
        Return None if no progression can be computed

        :param data: A dataframe containing all the sampled points so far
        :type data: pd.DataFrame | None
        :return: The current progression between 0.0 and 1.0, None otherwise
        :rtype: float
        """
        return None

    def dump(self, output_directory: Path | str) -> None:
        """
        Called at the end of the sampling process, to finalize the criterion if needed. This can
        be used to save
        the state of the criterion for example.

        Parameters
        ----------
        output_directory The directory where the criterion should save its state
        """
        raise NotImplementedError()


class TimeStoppingCriterion(StoppingCriterion):
    """
    A stopping criterion that stops the sampling process after a given amount of time
    This is criterion may not be exact, as it is based on the time it takes to sample a single
    point.

    This criterion will attempt to scale the number of samples taken on each iteration so that
    the sampling
    process takes approximately the given amount of time. This is done by estimating the time it
    takes to
    sample a single point, and then limiting the number of samples taken on each iteration to the
    amount of
    time left divided by the time it takes to sample a single point.

    >>> criterion = TimeStoppingCriterion(1)
    >>> criterion.init()
    >>> criterion.is_reached(pd.DataFrame(), pd.DataFrame())
    False
    >>> time.sleep(1.5)
    >>> criterion.is_reached(pd.DataFrame(), pd.DataFrame())
    True
    """

    def __init__(self, max_time_in_seconds: float):
        """

        Parameters
        ----------
        max_time_in_seconds The maximum amount of time in seconds that the sampling process can take
        """

        self.max_time = max_time_in_seconds
        self.time_start = None

    def init(self) -> None:
        """
        Reset the timer to the current time
        """
        self.time_start = time.time()

    def is_reached(self, data: pd.DataFrame, current_error: pd.DataFrame) -> bool:
        """
        Return true if the maximum amount of time has been reached. Note that this criterion may
        not be exact
        Parameters
        ----------
        data A dataframe containing all the sampled points so far
        current_error Unused in this criterion

        Returns
        -------
        True if the maximum amount of time has been reached, False otherwise
        """
        return time.time() - self.time_start > self.max_time

    def max_samples(self, data: pd.DataFrame | None) -> int:
        """
        Returns the maximum number of samples that can be taken on a given iteration, based on
        the amount of time
        taken so far and the time it takes to sample a single point

        Parameters
        ----------
        data A dataframe containing all the sampled points so far

        Returns
        -------
        -1 if there is no limit, or the maximum number of samples that can be taken on a given
        iteration otherwise

        """
        if data is None or len(data) == 0:
            return -1

        time_left = max(0, self.max_time - (time.time() - self.time_start))
        time_per_sample = (time.time() - self.time_start) / len(data)

        return int(time_left / time_per_sample)

    def get_progression(self, data: pd.DataFrame | None) -> float:
        """
        Compute the current percentage progression relative to this criterion

        :param data: A dataframe containing all the sampled points so far
        :type data: pd.DataFrame | None
        :return: The current progression between 0.0 and 1.0
        :rtype: float
        """
        return (time.time() - self.time_start) / self.max_time

    def dump(self, output_directory: Path | str):
        """
        Unused in this criterion
        """
        return


class MaxNSampleStoppingCriterion(StoppingCriterion):
    """
    Criterion that stops the sampling process after a given number of samples have been taken

    This criterion will attempt to scale the number of samples taken on each iteration so that
    the sampling
    process samples exactly the correct number of points.

    >>> criterion = MaxNSampleStoppingCriterion(10)
    >>> criterion.init()
    >>> criterion.is_reached(pd.DataFrame(), pd.DataFrame())
    False
    >>> data = pd.DataFrame(np.random.rand(5, 2), columns=["x", "y"])
    >>> criterion.is_reached(data, pd.DataFrame())
    False
    """

    def __init__(self, n_samples: int):
        """

        Parameters
        ----------
        n_samples The maximum number of samples allowed in the dataset
        """
        self.max_n_samples = n_samples

    def init(self) -> None:
        """
        Unused in this criterion
        """
        return

    def is_reached(self, data: pd.DataFrame, current_error: pd.DataFrame) -> bool:
        """
        Check if the maximum number of samples has been reached

        Parameters
        ----------
        data A dataframe containing all the sampled points so far
        current_error Unused in this criterion

        Returns
        -------
        True if the maximum number of samples has been reached, False otherwise
        """

        if data is None:
            return False
        return len(data) >= self.max_n_samples

    def max_samples(self, data: pd.DataFrame | None) -> int:
        """
        Return the total number of samples that can be taken on the next iteration to reach the
        maximum number of samples
        allowed

        Parameters
        ----------
        data A dataframe containing all the sampled points so far

        Returns
        -------
        -1 if there is no limit, or the maximum number of samples that can be taken on a given
        iteration otherwise
        """

        if data is None:
            return self.max_n_samples
        return max(0, self.max_n_samples - len(data))

    def get_progression(self, data: pd.DataFrame | None) -> float:
        """
        Compute the current percentage progression relative to this criterion

        :param data: A dataframe containing all the sampled points so far
        :type data: pd.DataFrame | None
        :return: The current progression between 0.0 and 1.0
        :rtype: float
        """
        return len(data) / self.max_n_samples

    def dump(self, output_directory: Path | str) -> None:
        """
        Unused in this criterion
        """
        return


class ErrorConvergenceStoppingCriterion(StoppingCriterion):
    """
    Criterion that stops the sampling process when the error on the last few iterations has
    converged

    This criterion uses a window of the last few iterations to measure the convergence of the
    error. If the variance
    of the error is below a given threshold, the criterion is considered reached.

    >>> criterion = ErrorConvergenceStoppingCriterion(0.1, 5)
    >>> criterion.init()
    >>> # Create a fake error dataframe with high variance
    >>> error = pd.DataFrame([i for i in range(10)], columns=["error"])
    >>> criterion.is_reached(pd.DataFrame(), error)
    False
    >>> # Create a fake error dataframe with low variance
    >>> error = pd.DataFrame([1] * 10, columns=["error"])
    >>> criterion.is_reached(pd.DataFrame(), error)
    True
    """

    def __init__(self, threshold: float, window_size: int = 5):
        """

        parameters
        ----------
        threshold
            The threshold for the variance of the error.
            The criterion is reached when the maximum variance of the errors is below this threshold
        window_size
            The size of the window used to compute the variance
            While the number of iteration is below this value, the criterion is not reached
        """

        if window_size < 2:
            raise ValueError("Window size must be at least 1")

        self.threshold = threshold
        self.window_size = window_size
        self.past_errors = None
        self.past_change = None

    def init(self):
        """
        Reset the internal state (errors and variance) of the criterion before starting a new run
        """
        self.past_errors = None
        self.past_change = None

    def is_reached(self, data: pd.DataFrame, current_error: pd.DataFrame) -> bool:
        """
        Check if the error has converged (i.e. the variance is below the defined threshold)

        Parameters
        ----------
        data
            Unused in this criterion
        current_error
            A dataframe containing the error on the current iteration

        Returns
        -------

        True if the error has converged, False otherwise
        """

        if self.past_errors is None:
            self.past_errors = current_error
            return False
        else:
            self.past_errors = pd.concat([self.past_errors, current_error], axis=0, ignore_index=True)

        if len(self.past_errors) < self.window_size:
            return False

        # Measure convergence since the last few iterations
        variances = abs(self.past_errors.iloc[-self.window_size :].var())

        if self.past_change is None:
            # First iteration, convert the Series to a DataFrame, and transpose it so there is
            # one column
            # per objective
            self.past_change = variances.to_frame().transpose()
        else:
            # Append the new row to the DataFrame
            self.past_change = pd.concat([self.past_change, variances], ignore_index=True)

        # We consider the maximum variance across all objectives as the convergence measurement
        max_convergence = variances.max()
        return bool(max_convergence < self.threshold)  # We need to convert to bool to avoid returning a numpy.bool_

    def max_samples(self, data: pd.DataFrame | None) -> int:
        """
        Unused in this criterion
        """
        return -1

    def dump(self, output_directory: Path | str):
        """
        Save the convergence data to a CSV file

        Parameters
        ----------
        output_directory
            The directory where the CSV file will be saved, as "error_convergence.csv

        Returns
        -------

        """
        if self.past_change is not None:
            self.past_change.to_csv(
                Path(output_directory) / "error_convergence.csv",
                index_label="iteration",
            )


class StoppingCriterionFactory:
    """
    Helper factory for creating stopping criterion from a dictionary, or a set of dictionaries
    """

    @staticmethod
    def create_all_from_dict(config: dict) -> list[StoppingCriterion]:
        """
        Create a list of stopping criterion from a dictionary containing multiple criterion

        The dictionary must have the following format:

        {
            "criterion_type_1": {
                "param_1": value_1,
            },
            "criterion_type_2": {
                ...
            }
        }

        Parameters
        ----------
        config
            The global configuration dictionary

        Returns
        -------
        Res
            A list of stopping criterion

        """
        return [StoppingCriterionFactory.create_from_dict(criterion_type, config[criterion_type]) for criterion_type in config]

    @staticmethod
    def create_from_dict(criterion_type: str, config: dict) -> StoppingCriterion:
        """
        Create a stopping criterion from a configuration dictionary

        Parameters
        ----------
        criterion_type
            The type of criterion to create

        config
            The dictionary containing the configuration parameters for the criterion

        Returns
        -------
        Res
            The created criterion if the type is recognized, otherwise a ValueError is raised
        """
        if criterion_type == "time":
            return TimeStoppingCriterion(**config)
        elif criterion_type == "max_n_samples":
            return MaxNSampleStoppingCriterion(**config)
        elif criterion_type == "error_convergence":
            return ErrorConvergenceStoppingCriterion(**config)
        else:
            raise ValueError(f"Unknown stopping criterion type: {criterion_type}")


def default_error_evaluator(samples: pd.DataFrame, features: list[str], objectives: list[str]) -> pd.DataFrame:
    """
    A default error evaluator for the adaptive sampling orchestrator
    This evaluator uses a XGBoost model to compute the MSE on the objectives

    >>> samples = pd.DataFrame([[1, 2, 3], [4, 5, 9]], columns=["a", "b", "c"])
    >>> features = ["a", "b"]
    >>> objectives = ["c"]
    >>> # Outputs a dataframe with the MSE for each objective, here, only the MSE for "c"
    >>> default_error_evaluator(samples, features, objectives)  # doctest: +ELLIPSIS
         c
    0  ...

    Parameters
    ----------
    samples
        The dataframe containing the data to evaluate

    features
        A list containing all the features to use for the evaluation

    objectives
        A list containing all the objectives to evaluate

    Returns
    -------
    A dataframe containing the MSE for each objective
    """

    from sklearn.metrics import mean_squared_error
    import lightgbm

    # We should use the same model as in the modeling phase
    model_generator = lightgbm.LGBMRegressor
    params = {"verbose": -1, "objective": "mae"}

    errors = []

    for objective in objectives:
        model = model_generator(**params)
        model.fit(samples[features], samples[objective])
        error = mean_squared_error(samples[objective], model.predict(samples[features]))
        errors.append(error)

    return pd.DataFrame(np.array(errors)[None], columns=objectives)


class AdaptiveSamplingOrchestrator:
    """
    A generic orchestrator to automate the adaptive sampling process
    This class handles initialization, execution, and stopping of the adaptive sampling
    and offers a simple API to customize the process. It is conceived to be the entry point
    for the adaptive sampling process.

    All the adaptive sampling algorithm can also be used manually instead, but this class simplifies
    the process and offers generic solutions for handling the stopping criterion and
    convergence evaluation.

    >>> from mlkaps.sampling.adaptive import AdaptiveSamplingOrchestrator, HVSampler
    >>> features = {"a": [0, 5], "b": [0, 5]}
    >>> sampler = HVSampler({"a": "int", "b": "int"}, features)
    >>> f = lambda df: pd.concat([df, df["a"] + df["b"]], axis=1)
    >>> stopping_criteria = [MaxNSampleStoppingCriterion(200)]
    >>> orchestrator = AdaptiveSamplingOrchestrator(features, f, sampler, None, stopping_criteria)
    >>> # Output a dataframe containing all the samples
    >>> orchestrator.run() # doctest: +ELLIPSIS
        a  b  0
    0   ...

    [200 rows x 3 columns]
    """

    def __init__(
        self,
        features: dict,
        execution_function: Callable,
        adaptive_sampler: AdaptiveSampler,
        output_directory: Path | str | None,
        stopping_criteria: list[StoppingCriterion] = None,
        error_evaluator: Callable[[pd.DataFrame, list, list], pd.DataFrame] = default_error_evaluator,
        n_samples_per_iteration: int = 100,
    ):
        """
        Create a new adaptive sampling orchestrator

        Parameters
        ----------
        features
            A dictionary containing the features to use for the adaptive sampling. The keys
            are the feature names, and the values are arrays of the features values

        execution_function
            A function that takes a dataframe as input, and returns a dataframe containing
            the original data, and extra columns containing the objectives values

        adaptive_sampler
            The adaptive sampler to use for the sampling process

        output_directory
            The directory where the adaptive sampling data will be saved
            If not set or None, the data will not be saved

        stopping_criteria
            A list of stopping criterion to use for stopping the adaptive sampling process
            If not set, a list of default criterion will be used:
                - ErrorConvergenceStoppingCriterion(0.002)
                    Stop when the variance on the error is below 0.002
                - TimeStoppingCriterion(600)
                    Stops after 600 seconds
        error_evaluator
            The function used to evaluate the modeling error based on the current samples
            If not set, a default evaluator will be used based on XGBoost and MSE

        n_samples_per_iteration
            The number of samples to generate per iteration
        """

        self.features = features

        self.execution_function = execution_function
        self.adaptive_sampler = adaptive_sampler

        if output_directory is None or output_directory == "":
            self.output_directory = None
        else:
            self.output_directory = Path(output_directory)

        self.criteria = stopping_criteria
        if self.criteria is None or len(self.criteria) == 0:
            self.criteria = [
                ErrorConvergenceStoppingCriterion(0.002),
                TimeStoppingCriterion(600),
            ]

        self.error_evaluator = error_evaluator
        self.n_samples_per_iteration = n_samples_per_iteration

        self._verify_arguments()

    def _verify_arguments(self):
        if self.features is None or len(self.features) == 0:
            raise ValueError("The features must be a non-empty dictionary")

        if self.adaptive_sampler is None:
            raise ValueError("Adaptive Sampler Orchestrator run on None sampler object")

        if self.execution_function is None:
            raise ValueError("Adaptive Sampler Orchestrator run on None execution function")

        if self.error_evaluator is None:
            raise ValueError("Adaptive Sampler Orchestrator run on None error evaluator")

        if self.n_samples_per_iteration is None or self.n_samples_per_iteration < 0:
            raise ValueError("Adaptive Sampler Orchestrator run with invalid number of samples per " "iteration")

    def run(self):
        """
        Run the adaptive sampling process, including initialization and finalization

        Returns
        -------
        pd.DataFrame
            The final dataset containing all samples
        """
        self._start()
        dataset = self._sample()
        self._dump()

        return dataset

    def _start(self) -> None:
        """
        Initialize all the components of the adaptive sampling process,
        including the sampler itself and all the criterion
        """

        self.adaptive_sampler.reset()
        for criterion in self.criteria:
            criterion.init()

    def _sample(self):
        """
        Run the adaptive sampler

        Returns
        -------
        pd.DataFrame
            The final dataset containing all samples
        """

        done = False
        dataset = None
        self.objectives = None

        if self.n_samples_per_iteration == 0:
            return None

        with tqdm(desc="Running adaptive sampling", unit=" percent", leave=None) as pbar:
            while not done:
                # find the maximum number of samples we can take for the next iteration
                n_samples = self._find_n_samples_next_iteration(dataset)

                # Run the adaptive sampler
                dataset = self.adaptive_sampler.sample(n_samples, dataset, self.execution_function)

                if self.output_directory is not None:
                    dataset.to_csv(self.output_directory / "samples.csv", index=False)

                # If this is the first iteration, we need to extract the objectives
                # From the results dataset
                if self.objectives is None:
                    self.objectives = [col for col in dataset.columns if col not in self.features]

                # Evaluate the error for each objective
                current_error = self.error_evaluator(dataset, list(self.features.keys()), self.objectives)

                # Check if we must stop
                done = self._check_any_criteria_reached(dataset, current_error)
                self._update_progress(dataset, pbar)

        return dataset

    def _dump(self):
        """
        Finalize the adaptive sampling process, including the sampler itself and all the criterion
        """

        if self.output_directory is None:
            return

        # Create the directories if they don't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.adaptive_sampler.dump(self.output_directory)

        for criterion in self.criteria:
            criterion.dump(self.output_directory)

    def _check_any_criteria_reached(self, dataset: pd.DataFrame, current_error: pd.DataFrame) -> bool:
        """
        Check if any of the stopping criterion has been reached

        Parameters
        ----------
        dataset
            The current dataset containing all points sampled so far

        current_error
            The current error for each objective

        Returns
        -------
        bool:
            True if any of the stopping criterion has been reached, False otherwise

        """
        for criterion in self.criteria:
            if criterion.is_reached(dataset, current_error):
                return True
        return False

    def _update_progress(self, dataset: pd.DataFrame, pbar: tqdm):
        """
        Try update the progression bar if possible
        """

        cmax = None
        for criterion in self.criteria:
            val = criterion.get_progression(dataset)
            if val is not None:
                cmax = val if cmax is None else max(cmax, val)

        if cmax is None:
            return

        # If we found a criterion that can give us the progression, then change the bar to display the progress as a percentage
        # Otherwise the bar will remain empty
        pbar.total = 100
        pbar.n = cmax * 100
        pbar.refresh()

    def _find_n_samples_next_iteration(self, dataset: pd.DataFrame) -> int:
        """
        Find the maximum number of samples we can take for the next iteration, based on the stopping
        criterion and used defined sampler per iteration

        Parameters
        ----------
        dataset
            The current dataset containing all points sampled so far

        Returns
        -------
        int:
            The maximum number of samples we can take for the next iteration
        """
        max_samples = self.n_samples_per_iteration
        for criterion in self.criteria:
            local_max_samples = criterion.max_samples(dataset)
            if local_max_samples >= 0:
                max_samples = min(max_samples, local_max_samples)
        return max_samples
