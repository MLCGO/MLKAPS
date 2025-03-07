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
import pandas as pd
import pytest


def _build_discard_sampler(functor=None, runner=None, pre_execution_callbacks=None):
    if functor is None:
        functor = lambda x: {"r": x["id"]}

    if runner is None:
        runner = MonoFunctionHarness(functor, expected_keys=["r"])

    resolver = DiscardResolver()
    sampler = MonoKernelExecutor(
        runner, resolver, pre_execution_callbacks=pre_execution_callbacks
    )

    return sampler


class TestMonoKernelSampler:

    def test_can_sample(self):
        sampler = _build_discard_sampler()
        samples = pd.DataFrame({"id": [1, 2, 3]})
        result = sampler(samples)

        assert result.equals(pd.DataFrame({"id": [1, 2, 3], "r": [1, 2, 3]}))

    def test_filters_failures(self):

        def fail_on_even(sample):
            if sample["id"] % 2 == 0:
                raise ValueError("Even id")
            return {"r": sample["id"]}

        sampler = _build_discard_sampler(functor=fail_on_even)

        # On Windows astype(int) converts int64 to int32
        # This is a Windows "feature" https://github.com/pandas-dev/pandas/issues/44925
        # We need to add astype(int) to the assert below
        samples = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        results = sampler(samples).astype(int)

        assert results.equals(
            pd.DataFrame({"id": [1, 3, 5, 7, 9], "r": [1, 3, 5, 7, 9]}).astype(int)
        )

    def test_handles_full_failure(self):
        def fail(sample):
            raise ValueError("Always fails")

        sampler = _build_discard_sampler(functor=fail)

        samples = pd.DataFrame({"id": [1, 2, 3, 4, 50]})
        results = sampler(samples)

        assert len(results) == 0
        assert all(results.columns == ["id", "r"])

    def test_raise_on_invalid_return(self):

        runner = lambda x: None

        sampler = _build_discard_sampler(runner=runner)
        samples = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(AttributeError):
            sampler(samples)

        runner = lambda x: {}

        sampler = _build_discard_sampler(runner=runner)
        samples = pd.DataFrame({"id": [1, 2, 3]})

        with pytest.raises(AttributeError):
            sampler(samples)

    def test_handles_no_samples(self):
        sampler = _build_discard_sampler()

        results = sampler(None)
        assert results is None

        results = sampler(pd.DataFrame([], columns=["id"]))
        assert results is None

    def test_can_use_pre_execution_callbacks(self):

        def callback():
            callback.ready = True

        def functor(samples):
            if not callback.ready:
                raise ValueError("Pre execution callback not called")
            return samples["id"]

        sampler = _build_discard_sampler(
            functor=functor, pre_execution_callbacks=[callback]
        )

        samples = pd.DataFrame({"id": [1, 2, 3]})
        sampler(samples)

    def test_can_use_multiple_pre_execution_callbacks(self):

        def callback():
            callback.ready = True

        def callback2():
            callback2.ready = True

        def functor(samples):
            if not callback.ready or not callback2.ready:
                raise ValueError("Pre execution callback not called")
            return samples["id"]

        sampler = _build_discard_sampler(
            functor=functor, pre_execution_callbacks=[callback, callback2]
        )

        samples = pd.DataFrame({"id": [1, 2, 3]})
        sampler(samples)
