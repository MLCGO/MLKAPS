"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.sample_collection.subprocess_harness import (
    ProcessCleanupHandler,
)
import pathlib
import pytest
import subprocess
import os
import sys
import psutil
import pandas as pd
from pandas.testing import assert_frame_equal


def get_samples_path():
    return pathlib.Path(__file__).parent / "samples.csv"


# This is a Windows-specific test to see if the process still exists.
def check_if_process_active(pid) -> bool:
    assert os.name == "nt"
    try:
        process = psutil.Process(pid)
    except psutil.Error:  # includes NoSuchProcess error
        return False
    if psutil.pid_exists(pid) and process.status() not in (
        psutil.STATUS_DEAD,
        psutil.STATUS_ZOMBIE,
    ):
        return True
    return False


class TestTimeoutRecovery:

    # we are going to run the dummy_recovery_kernel.  We will kill it with a timeout,
    # and then run it again.  This is to test that the samples checkpoint preserved
    # the state when we killed the kernel.

    def test_can_timeout(self):
        # setup_logging()
        # add_file_logger(pathlib.Path(__file__).parent)

        # note this has to match what is in dummy_recovery_kernel.py
        expected_results = pd.DataFrame(
            {
                "Y0": [3, 1, 3, 4, 3, 2, 1, 1, 3, 4, 4, 3, 2, 3, 2, 2, 2, 2, 3, 2],
                "m": [
                    304,
                    666,
                    376,
                    1464,
                    1310,
                    1101,
                    449,
                    159,
                    521,
                    1029,
                    1391,
                    86,
                    811,
                    231,
                    739,
                    956,
                    884,
                    1174,
                    1246,
                    594,
                ],
                "n": [
                    956,
                    1319,
                    376,
                    159,
                    1191,
                    811,
                    231,
                    1174,
                    1391,
                    739,
                    304,
                    594,
                    1029,
                    1464,
                    666,
                    1246,
                    521,
                    884,
                    86,
                    449,
                ],
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
                ],
            }
        )

        t_out = 1
        samples_path = get_samples_path()
        if samples_path.exists() and samples_path.is_file():
            samples_path.unlink()  # Remove the file

        tries = 0
        timed_out_cnt = 0
        while tries < 10:
            tries = tries + 1
            handler = ProcessCleanupHandler(timeout=t_out)
            arguments = [sys.executable, str(pathlib.Path(__file__).parent / "dummy_recovery_kernel.py")]
            res = handler.run(
                arguments,
                text=True,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            if res.timed_out:
                timed_out_cnt = timed_out_cnt + 1
            else:
                assert res.exitcode == 0, "not a clean exit"
                break

            if samples_path.exists() and samples_path.is_file():
                samples = pd.read_csv(samples_path)
                print(f"Samples file exists with {len(samples)} rows.")
            else:
                print("Samples file does not exist.")

        if samples_path.exists() and samples_path.is_file():
            results = pd.read_csv(samples_path)
            assert_frame_equal(results.sort_index(axis=1), expected_results.sort_index(axis=1))
        else:
            raise ValueError(f"Samples file {samples_path} does not exist.")

        assert timed_out_cnt > 0, "did not time out"

        # Check pid is not running anymore
        if os.name == "nt":
            assert not check_if_process_active(res.pid)
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(res.pid, 0)

        # clean up
        if samples_path.exists() and samples_path.is_file():
            samples_path.unlink()  # Remove the file
