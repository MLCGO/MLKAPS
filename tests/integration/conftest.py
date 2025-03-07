"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import matplotlib

matplotlib.use("Agg")
import builder_helper as helper  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="class")
def builder_helper():
    return helper.TestBuilderHelper()
