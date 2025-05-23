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
import pathlib
import time
import sys


# this dummy kernel is used in test_samples_recovery.py.  We want to kill the
# this kernel during its execution with a timeout, and then rerun it. On the rerun
# if will read the samples that have already been checkpointed, and restart after
# those.


def build_discard_sampler4(samples_checkpoint: SamplesCheckpoint, sleep_time: int):
    # special sampler for checkpoint4/ test_maybe_load_samples4
    functor_call_count = 0

    def checkpoint4_functor(x):
        nonlocal functor_call_count
        # Create a lookup table for 'm' to 'performance'
        m_to_performance = {
            304: [0.001414, 0],
            666: [0.008677, 1],
            376: [0.000959, 2],
            1464: [0.001800, 3],
            1310: [0.027301, 4],
            1101: [0.008646, 5],
            449: [0.000573, 6],
            159: [0.000461, 7],
            521: [0.005484, 8],
            1029: [0.007221, 9],
            1391: [0.002776, 10],
            86: [0.000213, 11],
            811: [0.008382, 12],
            231: [0.001397, 13],
            739: [0.004046, 14],
            956: [0.016034, 15],
            884: [0.004958, 16],
            1174: [0.009156, 17],
            1246: [0.000233, 18],
            594: [0.001868, 19],
        }

        perf, i = m_to_performance[x["m"]]
        # for debugging
        # log_file_path = pathlib.Path("checkpoint4_log.txt")
        # with log_file_path.open("a") as log_file:
        #     log_file.write(f"entry {i}\n")

        print(f"entry {i}")

        # we need sleep so that we can time out, but we only want to sleep once,
        # because otherwise we will always timeout while sleeping and never make progress
        # the batch size for writing samples out is 10, and we want to get some written
        # before we timeout.  So checking for sample 11 here.
        # ** note this should change if we change the batch size **
        if i == 11 and functor_call_count == 11:
            print("sleeping")
            time.sleep(sleep_time)
            print("waking")

        functor_call_count = functor_call_count + 1
        return {"performance": perf}

    functor = checkpoint4_functor
    runner = MonoFunctionHarness(functor, expected_keys=["performance"])
    resolver = DiscardResolver()
    sampler = MonoKernelExecutor(runner, resolver, samples_checkpoint)
    return sampler


def get_output_dir():
    return pathlib.Path(__file__).parent


def build_samples_checkpoint(parameters_type: dict, objectives: list):
    output_directory = get_output_dir()
    samples_checkpoint = SamplesCheckpoint(output_directory, parameters_type, objectives)
    return samples_checkpoint


def create_checkpoint4(sleep_time: int):
    # example that failed for Eric
    samples_checkpoint = build_samples_checkpoint(
        {
            "ratio_Z0_in_Z1_Zmax": "float",
            "Z1": "int",
            "X0": "int",
            "ratio_X1_in_X0_XYmax": "float",
            "Y0": "int",
            "ratio_Y1_in_Y0_XYmax": "float",
            "XY_s": "int",
            "Z_s": "int",
            "m": "int",
            "n": "int",
        },
        ["performance"],
    )
    sampler = build_discard_sampler4(samples_checkpoint, sleep_time)

    samples = pd.DataFrame(
        {
            "Y0": [3, 1, 3, 4, 3, 2, 1, 1, 3, 4, 4, 3, 2, 3, 2, 2, 2, 2, 3, 2],
            "m": [304, 666, 376, 1464, 1310, 1101, 449, 159, 521, 1029, 1391, 86, 811, 231, 739, 956, 884, 1174, 1246, 594],
            "n": [956, 1319, 376, 159, 1191, 811, 231, 1174, 1391, 739, 304, 594, 1029, 1464, 666, 1246, 521, 884, 86, 449],
            "Z1": [71, 84, 58, 97, 20, 135, 122, 224, 109, 237, 250, 46, 148, 186, 199, 173, 211, 7, 160, 33],
            "Z_s": [23, 33, 78, 98, 43, 58, 3, 93, 83, 28, 48, 53, 68, 8, 88, 18, 38, 63, 13, 73],
            "ratio_Y1_in_Y0_XYmax": [
                0.175,
                0.125,
                0.925,
                0.275,
                0.975,
                0.625,
                0.075,
                0.375,
                0.425,
                0.725,
                0.675,
                0.225,
                0.025,
                0.775,
                0.325,
                0.525,
                0.875,
                0.825,
                0.575,
                0.475,
            ],
            "ratio_Z0_in_Z1_Zmax": [
                0.875,
                0.925,
                0.525,
                0.125,
                0.975,
                0.075,
                0.375,
                0.675,
                0.475,
                0.175,
                0.825,
                0.025,
                0.625,
                0.775,
                0.225,
                0.575,
                0.325,
                0.725,
                0.425,
                0.275,
            ],
            "ratio_X1_in_X0_XYmax": [
                0.575,
                0.925,
                0.875,
                0.425,
                0.675,
                0.525,
                0.775,
                0.475,
                0.325,
                0.175,
                0.625,
                0.225,
                0.825,
                0.125,
                0.725,
                0.275,
                0.025,
                0.975,
                0.075,
                0.375,
            ],
            "XY_s": [43, 68, 98, 48, 58, 33, 38, 18, 8, 88, 53, 63, 93, 73, 3, 13, 28, 83, 78, 23],
            "X0": [1, 4, 2, 1, 3, 2, 2, 2, 4, 3, 4, 3, 3, 2, 2, 3, 3, 1, 3, 2],
        }
    )

    sample_count = len(samples)
    # see if we have some results already (from a previeous run)
    samples_reloaded = samples_checkpoint.maybe_load_samples()
    if samples_reloaded is not None:
        sample_count = sample_count - len(samples_reloaded)

    if sample_count > 0:
        new_results = sampler(samples.tail(sample_count))
        results = pd.concat([samples_reloaded, new_results], ignore_index=True)

    else:
        results = samples_reloaded

    # note the expected order is the parameters in alphabetical order, followed by objectives in alphabetical order

    expected_results = pd.concat(
        [
            samples.reindex(sorted(samples.columns), axis=1),
            pd.DataFrame(
                {
                    "performance": [
                        0.001414,
                        0.008677,
                        0.000959,
                        0.001800,
                        0.027301,
                        0.008646,
                        0.000573,
                        0.000461,
                        0.005484,
                        0.007221,
                        0.002776,
                        0.000213,
                        0.008382,
                        0.001397,
                        0.004046,
                        0.016034,
                        0.004958,
                        0.009156,
                        0.000233,
                        0.001868,
                    ]
                },
            ),
        ],
        axis=1,
    )

    # check data frame is correct and then check that the checkpoint is correct
    assert_frame_equal(results, expected_results)
    samples_checkpoint.consistency_check(expected_results)
    return samples_checkpoint, results


def test_maybe_load_samples4(sleep_time: int):
    samples_checkpoint, results = create_checkpoint4(sleep_time)
    loaded_samples = samples_checkpoint.maybe_load_samples()
    assert_frame_equal(loaded_samples, results)


# pass how long we should sleep.  Local and remote recovery requires
# different sleep amounts.
if __name__ == "__main__":
    sleep_time = 11
    if len(sys.argv) >= 2:
        try:
            sleep_time = int(sys.argv[1])
        except ValueError:
            pass

    test_maybe_load_samples4(sleep_time)
