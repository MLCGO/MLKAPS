"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
import pandas as pd
from mlkaps.sampling.adaptive import HVSampler


class TestHVSampler:

    def test_can_run_1d(self):
        features = {"a": [0, 5]}

        def f(df):
            return pd.concat([df, df.apply(lambda x: x.iloc[0], axis=1)], axis=1)

        sampler = HVSampler({"a": "float"}, features)
        data = sampler.sample(100, None, f)
        data = sampler.sample(200, data, f)
        assert data.shape == (300, 2)

    def test_can_run_2d(self):
        features = {"a": [0, 5], "b": [0, 5]}

        def f(df):
            return pd.concat([df, df.apply(lambda x: x.iloc[0] + x.iloc[1], axis=1)], axis=1)

        sampler = HVSampler({"a": "float", "b": "float"}, features)
        data = sampler.sample(100, None, f)
        data = sampler.sample(200, data, f)
        assert data.shape == (300, 3)

    def test_return_correct_n_samples_bootstrap(self):
        # Check that we return the correct number of samples, even when we use bootstrap
        # (We may return more samples than requested, for examples if the bootstrap samples >
        # n_samples)
        features = {"a": [0, 5]}

        def f(df):
            return pd.concat([df, df.apply(lambda x: x.iloc[0], axis=1)], axis=1)

        sampler = HVSampler({"a": "float"}, features)
        # Request a single sample, but use bootstrap
        data = sampler.sample(1, None, f)
        assert data.shape == (1, 2)

    def test_correctly_appends_to_data(self):
        features = {"a": [0, 100]}

        # Return 0 if x < 50, else return x
        def f(df):
            return pd.concat([df, df.apply(lambda x: x.iloc[0], axis=1)], axis=1)

        sampler = HVSampler({"a": "float"}, features)
        # Request a single sample, but use bootstrap
        data = None
        old_data = None
        for i in range(10):
            if i > 0:
                old_data = data
            data = sampler.sample(10, data, f)
            # Check that the old data is still there
            if i > 0:
                assert np.all(old_data == data[: old_data.shape[0]])
            assert data.shape == (10 * (i + 1), 2)

    def test_correctly_samples_high_variance_space(self):
        features = {"a": [0, 100]}

        # Return 0 if x < 50, else return x
        def f(df):
            return pd.concat([df, df.apply(lambda x: 0 if x.iloc[0] < 50 else x.iloc[0], axis=1)], axis=1)

        sampler = HVSampler({"a": "float"}, features)
        # Request a single sample, but use bootstrap
        data = None
        for _ in range(100):
            data = sampler.sample(10, data, f)
        # Check that most samples are in the high variance region
        sub_50 = np.sum(data["a"] < 50)

        # Less than 10% of the samples should be in the low variance region
        assert sub_50 < 100
