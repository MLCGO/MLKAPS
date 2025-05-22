"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.optimization.optimizer_checkpoint import OptimizerCheckpoint

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import pathlib


def _build_checkpoint():
    output_directory = pathlib.Path(__file__).parent
    optimizer_checkpoint = OptimizerCheckpoint(output_directory)
    # make sure sample.csv does not exist
    optimizer_checkpoint.delete_file()
    return optimizer_checkpoint


def _create_checkpoint():
    checkpoint = _build_checkpoint()
    samples = pd.DataFrame({"y": [-2.0, -2.0, 2.0, 2.0], "x": [-2.0, 2.0, -2.0, 2.0]})

    results = pd.DataFrame({"b": [2.7, 3.2, 2.7, 3.2], "y": [-2.0, -2.0, 2.0, 2.0], "x": [-2.0, 2.0, -2.0, 2.0]})

    results.to_csv(checkpoint.output_path, mode="a", header=True, index=False)
    return checkpoint, results, samples


class TestOptimizerCheckpoint:
    def test_maybe_load_samples(self):
        checkpoint, results, samples = _create_checkpoint()
        loaded_results = checkpoint.maybe_load_results(samples)
        assert_frame_equal(loaded_results, results)
        checkpoint.delete_file()

    def test_corrupt_checkpoint1(self):
        # more saved results than optimization points
        checkpoint, results, samples = _create_checkpoint()
        optimization_points = samples[:-2]
        with pytest.raises(AssertionError, match="more results"):
            checkpoint.maybe_load_results(optimization_points)
        checkpoint.delete_file()

    def test_corrupt_checkpoint2(self):
        # optimization points have different names than saved results
        checkpoint, results, samples = _create_checkpoint()
        optimization_points = samples.rename(columns={"x": "xxx"})
        with pytest.raises(AssertionError, match="not consistent"):
            checkpoint.maybe_load_results(optimization_points)
        checkpoint.delete_file()

    def test_corrupt_checkpoint3(self):
        # optimization points have different data than saved results
        checkpoint, results, samples = _create_checkpoint()
        optimization_points = samples
        optimization_points.loc[1, "x"] = 3.14
        with pytest.raises(AssertionError, match="not equal"):
            checkpoint.maybe_load_results(optimization_points)
        checkpoint.delete_file()

    def test_save(self):
        checkpoint, _, _ = _create_checkpoint()
        checkpoint.delete_file()

        samples = pd.DataFrame({"y": [-2.0, -2.0, 2.0, 2.0], "x": [-2.0, 2.0, -2.0, 2.0]})

        results = pd.DataFrame({"b": [2.7, 3.2, 2.7, 3.2], "y": [-2.0, -2.0, 2.0, 2.0], "x": [-2.0, 2.0, -2.0, 2.0]})

        for i in range(3):
            checkpoint.save(results.iloc[[i]])
            loaded_df = checkpoint.maybe_load_results(samples)
            assert_frame_equal(results.head(i + 1), loaded_df)
            checkpoint.consistency_check(results.head(i + 1))

        checkpoint.delete_file()
