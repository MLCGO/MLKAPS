"""
Copyright (C) 2020-2025 Intel Corporation
Copyright (C) 2022-2025 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
import pandas as pd
from pandas.testing import assert_frame_equal
import logging
import csv
import math


class OptimizerCheckpoint:
    def __init__(self, output_directory):

        # Capture path of output file.
        # Do not create the file here.  If it does exist, we will append new samples.
        # If it does not exist, we will create it when we first write to it.
        if isinstance(output_directory, str):
            output_directory = pathlib.Path(output_directory)

        self.output_path = output_directory / "optim.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.restarted = False

    def delete_file(self):
        # delete the checkpoint file and its directory
        if self.output_path.exists():
            self.output_path.unlink()

    def restart(self):
        # support quick restart
        logging.info("Quick-restart: Loading the optimization results from previos_run")
        if not self.output_path.exists():
            raise FileNotFoundError(self.output_path)
        self.restarted = True
        restart_samples = pd.read_csv(self.output_path)
        return restart_samples

    def maybe_load_results(self, optimization_points: pd.DataFrame, verbose: bool = True):
        # load results if they are there.  This is not for a quick restart

        # do not reload if we loaded them for a requested restart
        if self.restarted:
            return None

        # no reload if the file does not exist.
        if not self.output_path.exists():
            return None

        if verbose:
            logging.info(f"Found optimizations at '{self.output_path}', quick-restarting")
            logging.warning(
                f"Optimization will quick-restart by default, this is currently not configurable\n"
                f"Please delete '{self.output_path} to skip quick-restart."
            )
        loaded_results = pd.read_csv(self.output_path)
        # check if the loaded_samples seems correct
        self._sanity_check(loaded_results, optimization_points)

        return loaded_results

    def _sanity_check(self, saved_results: pd.DataFrame, optimization_points: pd.DataFrame):

        try:
            # if we have more saved results than requested optimization points, we probably have a user error.
            saved_len = len(saved_results)
            optim_len = len(optimization_points)
            assert saved_len <= optim_len, "We have more results in optim.csv than requested optimizaiton points"

            # Check that the names in the optimization points are in the saved results.
            saved_names = list(saved_results)
            optim_names = list(optimization_points)
            common_names = set(saved_names) & set(optim_names)
            assert optim_names, "Optimization points have no names"
            assert common_names == set(
                optim_names
            ), "Names of optimization points are not consistent with saved results in optim.csv"

            # Check that the data in the named columns match
            for name in optim_names:
                result_col = saved_results[name]
                sample_col = (optimization_points[name])[:saved_len]
                for rdata, sdata in zip(result_col, sample_col):
                    assert self._compare_with_tolerance(
                        rdata, sdata
                    ), "Data in optimization points is not equal to the saved results in optim.csv"

        except AssertionError as e:
            logging.warning("Saved optimization results at '{self.output_path}' do not seem correct for this experiment")
            logging.warning(str(e))
            raise e

    def _compare_with_tolerance(self, value1, value2, tolerance=1e-5):
        """
        Compares two values, using a tolerance if they are floats.

        Args:
            value1: The first value to compare.
            value2: The second value to compare.
            tolerance: The tolerance value for float comparison (default is 1e-5).

        Returns:
            True if the values are equal (or close enough if floats), False otherwise.
        """
        if isinstance(value1, float) or isinstance(value2, float):
            return math.isclose(value1, value2, rel_tol=tolerance)
        else:
            return value1 == value2

    def consistency_check(self, samples):
        # check if samples.csv matches the samples in memory
        saved_samples = pd.read_csv(self.output_path)
        assert (
            samples.shape == saved_samples.shape
        ), f"Samples have shape {samples.shape} and saved_samples have shape {saved_samples.shape}"
        if len(samples) == 0:
            return
        try:
            assert_frame_equal(samples, saved_samples, rtol=1e-5)
        except AssertionError as e:
            raise e

    def save(self, best_config: pd.DataFrame):
        # Append to output file
        #  *** This should be the only place we write samples to the optim.csv file ***

        # make sure we preserve column order.
        if self.output_path.exists() and self.output_path.is_file():
            with self.output_path.open("r") as file:
                csv_reader = csv.reader(file)
                column_headers = next(csv_reader)
                best_config = best_config[column_headers]
                best_config.to_csv(self.output_path, mode="a", header=False, index=False)
        else:
            best_config.to_csv(self.output_path, mode="a", header=True, index=False)

        return best_config
