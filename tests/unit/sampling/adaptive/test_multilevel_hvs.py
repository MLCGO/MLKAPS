"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pandas as pd
import pytest

from mlkaps.sampling.adaptive.multilevel_hvs import MultilevelHVS
from mlkaps.sampling import ValueSequence


def _run_simple_multilevel_hvs(features_types, features_values, levels):
    sampler = MultilevelHVS(levels, features_types, features_values)

    def f(df):
        return pd.concat([df, df.apply(lambda x: x.iloc[0], axis=1)], axis=1)

    data = sampler.sample(60, None, f)
    data = sampler.sample(10, data, f)

    assert len(data) == 70


class TestMultilevelHVS:
    def test_can_run1(self):
        features_types = {"a": "int", "b": "int"}
        features_values = {"a": ValueSequence(0, 5, 1), "b": ValueSequence(-5, 0, 1)}
        levels = [["a"], ["b"]]

        _run_simple_multilevel_hvs(features_types, features_values, levels)

    def test_can_run_one_level(self):
        features_types = {"a": "int"}
        features_values = {"a": ValueSequence(0, 5, 1)}
        levels = [["a"]]

        _run_simple_multilevel_hvs(features_types, features_values, levels)

    def test_can_run_multi_level(self):
        features_types = {"a": "int", "b": "int", "c": "int"}
        features_values = {"a": ValueSequence(0, 5, 1), "b": ValueSequence(-5, 0, 1), "c": ValueSequence(0, 100, 1)}
        levels = [["a"], ["b"], ["c"]]

        _run_simple_multilevel_hvs(features_types, features_values, levels)

    def test_can_run_multi_features_level(self):
        features_types = {"a": "int", "b": "int"}
        features_values = {"a": ValueSequence(0, 5, 1), "b": ValueSequence(-5, 0, 1)}
        levels = [["a"], ["b", "a"]]

        _run_simple_multilevel_hvs(features_types, features_values, levels)

    def test_validate_multilevel_hvs(self, tmp_path):

        features_types = {"x": "int", "optim": "int"}
        features_values = {"x": ValueSequence(0, 300, 1, type=int), "optim": ValueSequence(0, 200, 1, type=int)}
        levels = [["x"], ["optim"]]

        def eval_features(tab):
            if tab.iloc[0] < 100:
                if tab.iloc[1] < 100:
                    return 1
                else:
                    return 10
            elif tab.iloc[0] < 200:
                if tab.iloc[1] < 100:
                    return 10
                else:
                    return 1
            else:
                if tab.iloc[1] < 100:
                    return 1
                else:
                    return 10

        sampler = MultilevelHVS(levels, features_types, features_values)

        def f(df):
            return pd.concat([df, df.apply(eval_features, axis=1)], axis=1)

        data = sampler.sample(30, None, f)
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(5, 5, figsize=(50, 50))

        for i in range(25):
            ax = axs[i // 5, i % 5]
            data = sampler.sample(30, data, f)

            ax.scatter(data["x"], data["optim"], marker="x", c=data[0], cmap="Dark2")

            partitions = sampler.partitions
            for p in partitions:
                xbounds = p.axes["x"].get_sampling_bounds()
                obounds = p.axes["optim"].get_sampling_bounds()
                rect = plt.Rectangle(
                    (xbounds[0], obounds[0]),
                    xbounds[1] - xbounds[0],
                    obounds[1] - obounds[0],
                    color="red",
                    alpha=0.2,
                    lw=2,
                )
                ax.add_patch(rect)

        fig.savefig(tmp_path / "test.png")


if __name__ == "__main__":
    pytest.main()
