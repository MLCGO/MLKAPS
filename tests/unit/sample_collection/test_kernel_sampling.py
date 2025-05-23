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


def _build_discard_sampler(samples_checkpoint, functor=None, runner=None, pre_execution_callbacks=None):
    def default_functor(x):
        return {"r": x["id"]}

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


class TestMonoKernelSampler:

    def test_can_sample(self):
        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint)

        samples = pd.DataFrame({"id": [1, 2, 3]})
        result = sampler(samples)
        expected_result = pd.DataFrame({"id": [1, 2, 3], "r": [1, 2, 3]})

        # check data frame is correct and then heck that the checkpoint is correct
        assert_frame_equal(result, expected_result)
        samples_checkpoint.consistency_check(expected_result)
        samples_checkpoint.delete_file()

    def test_can_sample_twice(self):
        # This is testing that on each call to the sampler, the results are appended
        # in the output_path file
        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint)

        samples1 = pd.DataFrame({"id": [1, 2, 3]})
        result1 = sampler(samples1)
        expected_result1 = pd.DataFrame({"id": [1, 2, 3], "r": [1, 2, 3]})

        # check data frame is correct and then check that the checkpoint is correct
        assert_frame_equal(result1, expected_result1)
        samples_checkpoint.consistency_check(expected_result1)

        samples2 = pd.DataFrame({"id": [4, 5, 6]})
        result2 = sampler(samples2)
        expected_result2 = pd.DataFrame({"id": [4, 5, 6], "r": [4, 5, 6]})
        assert_frame_equal(result2, expected_result2)

        result_all = pd.concat([result1, result2]).reset_index(drop=True)
        expected_result_all = pd.concat([expected_result1, expected_result2]).reset_index(drop=True)
        assert_frame_equal(result_all, expected_result_all)
        samples_checkpoint.consistency_check(expected_result_all)
        samples_checkpoint.delete_file()

    def test_column_order(self):
        # Check that we preserve column order.
        # The order is determined when we save the batch in SamplesCheckpoint.save_batch
        #
        samples_checkpoint = _build_samples_checkpoint(
            {"id": "int", "a": "Categorical", "b": "Categorical", "c": "float"}, ["r"]
        )
        sampler = _build_discard_sampler(samples_checkpoint)

        samples1 = pd.DataFrame(
            {"id": [1, 2, 3], "a": ["one", "two", "three"], "b": ["four", "five", "six"], "c": [3.14, 1.01, 0.05]}
        )

        # reverse the columns in samples1
        samples2 = samples1[samples1.columns[::-1]]

        # samples wiht different column orders, but same data, will give the same results.
        # this is because the samples with their results put their columns in a canonical order.
        result1 = sampler(samples1)
        result2 = sampler(samples2)
        assert_frame_equal(result1, result2)
        samples_checkpoint.delete_file()

    def test_filters_failures(self):

        def fail_on_even(sample):
            if sample["id"] % 2 == 0:
                raise ValueError("Even id")
            return {"r": sample["id"]}

        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint, functor=fail_on_even)

        samples = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = sampler(samples)

        # The value error in fail_on_even will put a NaN in the dataframe, which is later
        # deleted.  But it will cause the entire result column to be floating point.
        expected_result = pd.DataFrame({"id": [1, 3, 5, 7, 9], "r": [1.0, 3.0, 5.0, 7.0, 9.0]})
        assert_frame_equal(result, expected_result)
        samples_checkpoint.consistency_check(expected_result)
        samples_checkpoint.delete_file()

    def test_handles_full_failure(self):
        def fail(sample):
            raise ValueError("Always fails")

        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint, functor=fail)

        # the result should be an empty dataframe
        samples = pd.DataFrame({"id": [1, 2, 3, 4, 50]})
        result = sampler(samples)
        assert len(result) == 0

        expected_result = pd.DataFrame({"id": pd.Series(dtype="int"), "r": pd.Series(dtype="float")})
        assert_frame_equal(result, expected_result)
        samples_checkpoint.consistency_check(expected_result)
        samples_checkpoint.delete_file()

    def test_raise_on_invalid_return(self):
        def runner(x):
            return None

        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint, runner=runner)

        samples = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(AttributeError):
            sampler(samples)
        samples_checkpoint.delete_file()

    def test_handles_no_samples(self):
        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint)

        results = sampler(None)
        assert results is None

        results = sampler(pd.DataFrame([], columns=["id"]))
        assert results is None
        samples_checkpoint.delete_file()

    def test_can_use_pre_execution_callbacks(self):
        def callback():
            callback.ready = True

        def functor(samples):
            if not callback.ready:
                raise ValueError("Pre execution callback not called")
            return {"r": samples["id"]}

        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint, functor=functor, pre_execution_callbacks=[callback])

        samples = pd.DataFrame({"id": [1, 2, 3]})
        result = sampler(samples)

        expected_result = pd.DataFrame({"id": [1, 2, 3], "r": [1, 2, 3]})
        assert_frame_equal(result, expected_result)
        samples_checkpoint.consistency_check(expected_result)
        samples_checkpoint.delete_file()

    def test_can_use_multiple_pre_execution_callbacks(self):

        def callback():
            callback.ready = True

        def callback2():
            callback2.ready = True

        def functor(samples):
            if not callback.ready or not callback2.ready:
                raise ValueError("Pre execution callback not called")
            return {"r": samples["id"]}

        samples_checkpoint = _build_samples_checkpoint({"id": "int"}, ["r"])
        sampler = _build_discard_sampler(samples_checkpoint, functor=functor, pre_execution_callbacks=[callback, callback2])

        samples = pd.DataFrame({"id": [1, 2, 3]})
        result = sampler(samples)

        expected_result = pd.DataFrame({"id": [1, 2, 3], "r": [1, 2, 3]})
        assert_frame_equal(result, expected_result)
        samples_checkpoint.consistency_check(expected_result)
        samples_checkpoint.delete_file()
