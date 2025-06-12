"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.sample_collection.mono_kernel_executor import (
    MonoKernelExecutor,
)

from mlkaps.sample_collection.function_harness import MonoFunctionHarness
from mlkaps.sample_collection.failed_run_resolver import DiscardResolver
from mlkaps.sample_collection.samples_checkpoint import SamplesCheckpoint

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import pathlib
import numpy as np


def _build_discard_sampler4(samples_checkpoint):
    # special sampler for checkpoint4/ test_maybe_load_samples4

    def checkpoint4_functor(x):
        # Create a lookup table for 'm' to 'performance'
        m_to_performance = {
            304: 0.001414,
            666: 0.008677,
            376: 0.000959,
            1464: 0.001800,
            1310: 0.027301,
            1101: 0.008646,
            449: 0.000573,
            159: 0.000461,
            521: 0.005484,
            1029: 0.007221,
            1391: 0.002776,
            86: 0.000213,
            811: 0.008382,
            231: 0.001397,
            739: 0.004046,
            956: 0.016034,
            884: 0.004958,
            1174: 0.009156,
            1246: 0.000233,
            594: 0.001868,
        }
        return {"performance": m_to_performance[x["m"]]}

    functor = checkpoint4_functor
    runner = MonoFunctionHarness(functor, expected_keys=["performance"])
    resolver = DiscardResolver()
    sampler = MonoKernelExecutor(runner, resolver, samples_checkpoint)
    return sampler


def _build_discard_sampler(samples_checkpoint, functor=None, runner=None, pre_execution_callbacks=None):
    def default_functor(x):
        return {"r": x["id"] + 0.1}

    if functor is None:
        functor = default_functor

    if runner is None:
        runner = MonoFunctionHarness(functor, expected_keys=["r"])

    resolver = DiscardResolver()
    sampler = MonoKernelExecutor(runner, resolver, samples_checkpoint, pre_execution_callbacks=pre_execution_callbacks)
    return sampler


def _build_samples_checkpoint(parameters_type: dict, objectives: list):
    output_directory = pathlib.Path(__file__).parent
    samples_checkpoint = SamplesCheckpoint(output_directory, parameters_type, objectives)
    # make sure sample.csv does not exist
    samples_checkpoint.delete_file()
    return samples_checkpoint


def _create_checkpoint():
    samples_checkpoint = _build_samples_checkpoint({"id": "float"}, ["r"])
    sampler = _build_discard_sampler(samples_checkpoint)

    samples = pd.DataFrame({"id": [1.0, 2.0, 3.0]})
    results = sampler(samples)

    # note the expected order is the parameters in alphabetical order,  followed by objectives in alphabetical order
    expected_results = pd.DataFrame({"id": [1.0, 2.0, 3.0], "r": [1.1, 2.1, 3.1]})

    # check data frame is correct and then check that the checkpoint is correct
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


def _create_checkpoint1():
    # sample with both int and float columns
    samples_checkpoint = _build_samples_checkpoint({"id": "float", "j": "int"}, ["r"])
    sampler = _build_discard_sampler(samples_checkpoint)

    samples = pd.DataFrame({"id": [1.0, 2.0, 3.0], "j": [4, 5, 6]})
    results = sampler(samples)

    # note the expected order is the parameters in alphabetical order,  followed by objectives in alphabetical order
    expected_results = pd.DataFrame({"id": [1.0, 2.0, 3.0], "j": [4, 5, 6], "r": [1.1, 2.1, 3.1]})

    # check data frame is correct and then check that the checkpoint is correct
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


def _create_checkpoint2():
    # sample with int, float, and categorical columns
    samples_checkpoint = _build_samples_checkpoint({"id": "float", "j": "int", "k": "Categorical"}, ["r"])
    sampler = _build_discard_sampler(samples_checkpoint)

    samples = pd.DataFrame({"id": [1.0, 2.0, 3.0], "j": [4, 5, 6], "k": ["One", "Two", "Three"]})
    results = sampler(samples)

    # note the expected order is the parameters in alphabetical order,  followed by objectives in alphabetical order
    expected_results = pd.DataFrame(
        {"id": [1.0, 2.0, 3.0], "j": [4, 5, 6], "k": ["One", "Two", "Three"], "r": [1.1, 2.1, 3.1]}
    )

    # check data frame is correct and then check that the checkpoint is correct
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


def _create_checkpoint3():
    # sample with int, float, categorical, and boolean columns
    samples_checkpoint = _build_samples_checkpoint({"id": "float", "j": "int", "k": "Categorical", "b": "boolean"}, ["r"])
    sampler = _build_discard_sampler(samples_checkpoint)

    samples = pd.DataFrame({"id": [1.0, 2.0, 3.0], "j": [4, 5, 6], "k": ["One", "Two", "Three"], "b": [True, False, False]})
    results = sampler(samples)

    # note the expected order is the parameters in alphabetical order,  followed by objectives in alphabetical order
    expected_results = pd.DataFrame(
        {"b": [True, False, False], "id": [1.0, 2.0, 3.0], "j": [4, 5, 6], "k": ["One", "Two", "Three"], "r": [1.1, 2.1, 3.1]}
    )

    # check data frame is correct and then check that the checkpoint is correct
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


def _create_checkpoint4():
    # example that failed for Eric
    samples_checkpoint = _build_samples_checkpoint(
        {
            "ratio_Z0_in_Z1_Zmax": "float",
            "Z1": "int",
            "X0": "int",
            "ratio_X1_in_X0_XYmax": "float",
            "Y0": "int",
            "ratio_Y1_in_Y0_XYmax": "float",
            "XY_s": "int",
            "Z_s": "int",
            "m": "int",
            "n": "int",
        },
        ["performance"],
    )
    sampler = _build_discard_sampler4(samples_checkpoint)

    samples = pd.DataFrame(
        {
            "Y0": [3, 1, 3, 4, 3, 2, 1, 1, 3, 4, 4, 3, 2, 3, 2, 2, 2, 2, 3, 2],
            "m": [304, 666, 376, 1464, 1310, 1101, 449, 159, 521, 1029, 1391, 86, 811, 231, 739, 956, 884, 1174, 1246, 594],
            "n": [956, 1319, 376, 159, 1191, 811, 231, 1174, 1391, 739, 304, 594, 1029, 1464, 666, 1246, 521, 884, 86, 449],
            "Z1": [71, 84, 58, 97, 20, 135, 122, 224, 109, 237, 250, 46, 148, 186, 199, 173, 211, 7, 160, 33],
            "Z_s": [23, 33, 78, 98, 43, 58, 3, 93, 83, 28, 48, 53, 68, 8, 88, 18, 38, 63, 13, 73],
            "ratio_Y1_in_Y0_XYmax": [
                0.175,
                0.125,
                0.925,
                0.275,
                0.975,
                0.625,
                0.075,
                0.375,
                0.425,
                0.725,
                0.675,
                0.225,
                0.025,
                0.775,
                0.325,
                0.525,
                0.875,
                0.825,
                0.575,
                0.475,
            ],
            "ratio_Z0_in_Z1_Zmax": [
                0.875,
                0.925,
                0.525,
                0.125,
                0.975,
                0.075,
                0.375,
                0.675,
                0.475,
                0.175,
                0.825,
                0.025,
                0.625,
                0.775,
                0.225,
                0.575,
                0.325,
                0.725,
                0.425,
                0.275,
            ],
            "ratio_X1_in_X0_XYmax": [
                0.575,
                0.925,
                0.875,
                0.425,
                0.675,
                0.525,
                0.775,
                0.475,
                0.325,
                0.175,
                0.625,
                0.225,
                0.825,
                0.125,
                0.725,
                0.275,
                0.025,
                0.975,
                0.075,
                0.375,
            ],
            "XY_s": [43, 68, 98, 48, 58, 33, 38, 18, 8, 88, 53, 63, 93, 73, 3, 13, 28, 83, 78, 23],
            "X0": [1, 4, 2, 1, 3, 2, 2, 2, 4, 3, 4, 3, 3, 2, 2, 3, 3, 1, 3, 2],
        }
    )

    results = sampler(samples)

    # note the expected order is the parameters in alphabetical order,
    # followed by objectives in alphabetical order
    expected_results = pd.concat(
        [
            samples.reindex(sorted(samples.columns), axis=1),
            pd.DataFrame(
                {
                    "performance": [
                        0.001414,
                        0.008677,
                        0.000959,
                        0.001800,
                        0.027301,
                        0.008646,
                        0.000573,
                        0.000461,
                        0.005484,
                        0.007221,
                        0.002776,
                        0.000213,
                        0.008382,
                        0.001397,
                        0.004046,
                        0.016034,
                        0.004958,
                        0.009156,
                        0.000233,
                        0.001868,
                    ]
                }
            ),
        ],
        axis=1,
    )

    # note there are three copies of "results" which should all be equal.
    #   1) expected_results: this is the dataframe constructed in this function
    #   2) results: this is the dataframe returned by the call to the sampler above
    #   3) the saved results saved in the checkpoint when we called the sampler
    # The assertion below checks that 1) and 2) are equal
    # The call to consistency_check checks that 1) and 3) are equal
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


class TestSamplesCheckpoint:
    def test_maybe_load_samples(self):
        samples_checkpoint, results = _create_checkpoint()
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, results)
        samples_checkpoint.delete_file()

    def test_maybe_load_samples1(self):
        samples_checkpoint, results = _create_checkpoint1()
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, results)
        samples_checkpoint.delete_file()

    def test_maybe_load_samples2(self):
        samples_checkpoint, results = _create_checkpoint2()
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, results)
        samples_checkpoint.delete_file()

    def test_maybe_load_samples3(self):
        samples_checkpoint, results = _create_checkpoint3()
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, results)
        samples_checkpoint.delete_file()

    def test_maybe_load_samples4(self):
        samples_checkpoint, results = _create_checkpoint4()
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, results)
        samples_checkpoint.delete_file()

    def test_corrupt_checkpoint1(self):
        # drop a column from the checkpoint file
        samples_checkpoint, results = _create_checkpoint()
        check_df = pd.read_csv(samples_checkpoint.output_path)
        check_df.drop("id", axis=1, inplace=True)
        check_df.to_csv(samples_checkpoint.output_path, mode="w", header=True, index=False)
        with pytest.raises(AssertionError, match="columns, expected"):
            samples_checkpoint.maybe_load_samples()
        samples_checkpoint.delete_file()

    def test_corrupt_checkpoint2(self):
        # change name of a column
        samples_checkpoint, results = _create_checkpoint()
        check_df = pd.read_csv(samples_checkpoint.output_path)
        check_df = check_df.rename(columns={"id": "xxx"})
        check_df.to_csv(samples_checkpoint.output_path, mode="w", header=True, index=False)
        with pytest.raises(AssertionError, match="expected column names"):
            samples_checkpoint.maybe_load_samples()
        samples_checkpoint.delete_file()

    def test_corrupt_checkpoint3(self):
        # change type of a column
        samples_checkpoint, results = _create_checkpoint()
        check_df = pd.read_csv(samples_checkpoint.output_path)
        check_df["id"] = check_df["id"].astype(int)
        check_df.to_csv(samples_checkpoint.output_path, mode="w", header=True, index=False)
        with pytest.raises(AssertionError, match="not compatible with"):
            samples_checkpoint.maybe_load_samples()
        samples_checkpoint.delete_file()

    def test_corrupt_checkpoint4(self):
        # change type of a data element
        samples_checkpoint, results = _create_checkpoint()
        check_df = pd.read_csv(samples_checkpoint.output_path)
        check_df.loc[1, "id"] = "pi"  # note this causes a warning from pandas
        check_df.to_csv(samples_checkpoint.output_path, mode="w", header=True, index=False)
        with pytest.raises(AssertionError, match="not compatible with"):
            samples_checkpoint.maybe_load_samples()
        samples_checkpoint.delete_file()

    def test_corrupt_checkpoint5(self):
        # corrupt the checkpoint file with a NaN
        samples_checkpoint, results = _create_checkpoint()
        check_df = pd.read_csv(samples_checkpoint.output_path)
        check_df.loc[1, "r"] = np.nan
        check_df.to_csv(samples_checkpoint.output_path, mode="w", header=True, index=False)
        with pytest.raises(AssertionError, match="NaN value"):
            samples_checkpoint.maybe_load_samples()
        samples_checkpoint.delete_file()

    def test_save_batch(self):
        samples_checkpoint, results = _create_checkpoint()
        new_batch = pd.DataFrame({"id": [4.0, 5.0, 6.0], "r": [4.1, 5.1, 6.1]})
        new_batch = samples_checkpoint.save_batch(new_batch)
        new_data = pd.concat([results, new_batch], ignore_index=True)
        # load the checkpoint
        loaded_samples = samples_checkpoint.maybe_load_samples()
        assert_frame_equal(loaded_samples, new_data)
        samples_checkpoint.delete_file()
