"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import time

import pandas as pd
import pytest

from mlkaps.sampling.adaptive import (
    TimeStoppingCriterion,
    MaxNSampleStoppingCriterion,
    ErrorConvergenceStoppingCriterion,
)


class TestTimeCriterion:

    def test_correctly_reaches(self):
        criterion = TimeStoppingCriterion(0.1)

        criterion.init()
        time.sleep(0.2)
        assert criterion.is_reached(pd.DataFrame(), pd.DataFrame())

    def test_not_reached(self):
        criterion = TimeStoppingCriterion(0.5)
        criterion.init()

        assert not criterion.is_reached(pd.DataFrame(), pd.DataFrame())
        time.sleep(0.1)
        assert not criterion.is_reached(pd.DataFrame(), pd.DataFrame())

    def test_limits_number_of_samples(self):
        criterion = TimeStoppingCriterion(0.5)
        criterion.init()

        assert criterion.max_samples(None) == -1
        assert criterion.max_samples(pd.DataFrame()) == -1

        # Fake data with 10 samples
        fake_data = pd.DataFrame()
        fake_data["test"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # FIXME: This testing method is pretty bad, since if any delays occur, we might have
        # occasional failures

        criterion.init()
        # Wait for half of the time to pass
        time.sleep(0.25)
        # We should be able to sample atleast 7 more samples
        assert criterion.max_samples(fake_data) >= 7

        time.sleep(0.3)
        assert criterion.max_samples(fake_data) == 0


class TestNMaxSampleCriterion:
    def test_empty_data(self):
        criterion = MaxNSampleStoppingCriterion(10)
        criterion.init()

        assert not criterion.is_reached(None, None)
        assert not criterion.is_reached(pd.DataFrame(), pd.DataFrame())
        assert criterion.max_samples(None) == 10
        assert criterion.max_samples(pd.DataFrame()) == 10

    def test_enough_samples(self):
        criterion = MaxNSampleStoppingCriterion(10)
        criterion.init()

        fake_data = pd.DataFrame()
        # Fake data with 10 samples
        fake_data["test"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert criterion.is_reached(fake_data, pd.DataFrame())
        assert criterion.max_samples(fake_data) == 0

    def test_not_enough_samples(self):
        criterion = MaxNSampleStoppingCriterion(10)
        criterion.init()

        fake_data = pd.DataFrame()
        # Fake data with 5 samples
        fake_data["test"] = [0, 1, 2, 3, 4]
        assert not criterion.is_reached(fake_data, pd.DataFrame())
        assert criterion.max_samples(fake_data) == 5


class TestConvergenceCriterion:
    def test_empty_data(self):
        # Stop when the variance of the error is below 0.5
        criterion = ErrorConvergenceStoppingCriterion(0.5)
        criterion.init()

        assert not criterion.is_reached(pd.DataFrame(), pd.DataFrame())
        assert not criterion.is_reached(None, None)

    def test_never_limit_samples(self):
        # Stop when the variance of the error is below 0.5
        criterion = ErrorConvergenceStoppingCriterion(0.5, window_size=2)
        criterion.init()

        assert criterion.max_samples(None) == -1

        # Convergence = false
        fake_data = pd.DataFrame()
        fake_data["test"] = [0]
        assert criterion.max_samples(fake_data) == -1

        # Convergence = true
        fake_data = pd.DataFrame()
        fake_data["test"] = [0, 0]
        assert criterion.max_samples(fake_data) == -1

        criterion = ErrorConvergenceStoppingCriterion(0.5, window_size=5)
        criterion.init()

        # Not enough data
        fake_data = pd.DataFrame()
        fake_data["test"] = [0, 0, 0, 0]
        assert criterion.max_samples(fake_data) == -1

    def test_convergence(self):
        # Stop when the variance of the error is below 0.5
        criterion = ErrorConvergenceStoppingCriterion(0.5, window_size=10)
        criterion.init()

        # Check that the convergence is reached when the variance over 10 iterations is below 0.5 (0 here)
        fake_data = pd.DataFrame()
        fake_data["test"] = [0]
        for i in range(10):
            criterion.is_reached(fake_data, fake_data)

        assert criterion.is_reached(fake_data, fake_data)

    def test_no_convergence(self):
        # Stop when the variance of the error is below 0.5
        criterion = ErrorConvergenceStoppingCriterion(0.5, window_size=10)
        criterion.init()

        # Check that the convergence is NOT reached when the variance over 10 iterations is over 0.5
        fake_data = pd.DataFrame()
        for i in range(10):
            fake_data["test"] = [i]
            criterion.is_reached(fake_data, fake_data)

        assert not criterion.is_reached(fake_data, pd.DataFrame())

    def test_convergence_invalid_window_size(self):
        # Check that the window size is at least 2
        with pytest.raises(ValueError):
            ErrorConvergenceStoppingCriterion(0.5, window_size=1)

    def test_no_convergence_window_size(self):
        # Check that convergence is not reached when the window size is too small
        # Even if the variance is below the threshold
        criterion = ErrorConvergenceStoppingCriterion(0.5, window_size=10)
        criterion.init()

        # Check that the convergence is reached when the variance over 10 iterations is below 0.5 (0 here)
        fake_data = pd.DataFrame()
        fake_data["test"] = [0]
        for i in range(5):
            criterion.is_reached(fake_data, fake_data)

        assert not criterion.is_reached(fake_data, fake_data)
