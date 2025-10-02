"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""


def my_sqrt(sample):
    return {"r": sample["id"] ** 0.5}


def return_id(sample):
    return {"r": sample["id"]}


global_var = 42


def test_can_init_global(sample):
    return {"r": global_var}
