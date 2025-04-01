#!/bin/env python3
"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import sys
import os
import time


def main():

    if len(sys.argv) != 2:
        print("Usage: python dummy_kernel.py id")
        sys.exit(1)

    if os.environ.get("DO_SLEEP") == "1":
        import time

        time.sleep(3)

    n_output = 1

    if os.environ.get("N_OUTPUT") is not None:
        n_output = int(os.environ.get("N_OUTPUT"))
        print(n_output)

    print("Hello, World!")

    if n_output < 1:
        return

    id = int(sys.argv[1])
    print(id, end="")

    for i in range(n_output - 1):
        print(",", id / (i + 2), end="")


def functor(sample):
    # Take five second if id is below 5
    if sample["id"] < 5:
        time.sleep(2)
    return {"r": sample["id"]}


if __name__ == "__main__":
    main()
