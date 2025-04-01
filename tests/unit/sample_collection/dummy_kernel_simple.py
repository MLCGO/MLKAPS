#!/bin/env python3
"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""


def functor(sample):
    # Take five second if id is below 5
    # if sample["id"] < 5:
    #     time.sleep(5)
    return {"r": sample["id"]}


if __name__ == "__main__":
    samples = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 6}, {"id": 7}]
    for s in samples:
        result = functor(s)
        print("result:", result)
        print(s, result)
