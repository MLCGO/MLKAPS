"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.sample_collection.subprocess_harness import ProcessCleanupHandler, MonoSubprocessHarness
import time
import math
import pathlib
import pytest
import subprocess
import os
import shutil
import sys
import psutil

def temporary_env(func):
    def wrapper(*args, **kwargs):
        old_env = dict(os.environ)
        try:
            func(*args, **kwargs)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
    return wrapper

# On Windows, we add a python.exe to start of the command line when running a python program.
# We need to remove this when doing the check required in the tests below.
def check_returned_arguments(res_args, call_args):
    if os.name == "nt":
        if (len(res_args) == len(call_args) + 1 and res_args[0] == sys.executable):
            return res_args[1:] == call_args  
        else:
            return False
    else:
        return res_args == call_args 
    
# This is a Windows-specific test to see if the process still exists.
def check_if_process_active(pid: int or str) -> bool:
    assert(os.name == "nt")
    try:
        process = psutil.Process(pid)
    except psutil.Error as error:  # includes NoSuchProcess error
        return False
    if psutil.pid_exists(pid) and process.status() not in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE):
        return True
    return False   


class TestProcessCleanupHandler:
    def test_can_run(self):
        handler = ProcessCleanupHandler()
        arguments = [str(pathlib.Path(__file__).parent / "dummy_kernel.py"), "5"]
        res = handler.run(
            arguments,
            text=True,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        assert res.exitcode == 0
        assert check_returned_arguments(res.arguments,arguments)
        assert res.timed_out == False
        assert res.stdout == "Hello, World!\n5"

    def test_handles_unkown_file(self):
        handler = ProcessCleanupHandler()
        # Missing argument
        arguments = ["this_kernel_doesnt_exists.py"]

        with pytest.raises(FileNotFoundError):
            _ = handler.run(
                arguments,
                text=True,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )


    @temporary_env
    def test_can_timeout(self):
        if os.name == "nt":
            t_out = 1
        else:
            t_out = 0.2
        handler = ProcessCleanupHandler(timeout=t_out)
        os.environ["DO_SLEEP"] = "1"
        arguments = [str(pathlib.Path(__file__).parent / "dummy_kernel.py"), "5"]
        begin = time.time()
        res = handler.run(
            arguments,
            text=True,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        end = time.time()
        print(f"subprocess harness begin: {begin}, end: {end}, t_out: {t_out}")
        assert (end - begin) < 2*t_out
        if os.name == "nt":
            assert res.exitcode == 1
        else:
            assert res.exitcode == -15
        assert check_returned_arguments(res.arguments, arguments)
        assert res.timed_out == True
            
        # Check pid is not running anymore
        if os.name == "nt":
            assert not check_if_process_active(res.pid)
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(res.pid, 0)


class TestMonoSubprocessRunner:


    def test_can_sample(self):
        runner = MonoSubprocessHarness(
            objectives=["r"],
            objectives_bounds={"r": 42},
            executable_path=pathlib.Path(__file__).parent / "dummy_kernel.py",
            arguments_order=["id"],
        )

        res = runner({"id": 5})
        
        assert res.data == {"r": 5.0}
        assert res.error is None
        assert res.timed_out == False

    @temporary_env
    def test_can_sample_multiple(self):

        os.environ["N_OUTPUT"] = "2"
            
        runner = MonoSubprocessHarness(
            objectives=["r", "r2"],
            objectives_bounds={"r": 42, "r2": 81},
            executable_path=pathlib.Path(__file__).parent / "dummy_kernel.py",
            arguments_order=["id"],
        )

        res = runner({"id": 5})
        
        assert res.data == {"r": 5.0, "r2": 2.5}
        assert res.error is None
        assert res.timed_out == False

    @temporary_env
    def test_detect_invalid_return(self):
        
        os.environ["N_OUTPUT"] = "2"

        runner = MonoSubprocessHarness(
            objectives=["r"],
            objectives_bounds={"r": 42},
            executable_path=pathlib.Path(__file__).parent / "dummy_kernel.py",
            arguments_order=["id"],
        )

        res = runner({"id": 5})
        assert res.error is not None
        assert res.timed_out == False

        os.environ["N_OUTPUT"] = "1"
            
        runner = MonoSubprocessHarness(
            objectives=["r", "r2"],
            objectives_bounds={"r": 42, "r2": 81},
            executable_path=pathlib.Path(__file__).parent / "dummy_kernel.py",
            arguments_order=["id"],
        )

        res = runner({"id": 5})
        assert res.error is not None
        assert res.timed_out == False

        os.environ["N_OUTPUT"] = "0"
            
        runner = MonoSubprocessHarness(
            objectives=["r"],
            objectives_bounds={"r": 42},
            executable_path=pathlib.Path(__file__).parent / "dummy_kernel.py",
            arguments_order=["id"],
        )

        res = runner({"id": 5})
        assert res.error is not None
        assert res.timed_out == False

    def test_handles_invalid_kernel(self):

        runner = MonoSubprocessHarness(
            objectives=["r"],
            objectives_bounds={"r": 42},
            executable_path=pathlib.Path(__file__).parent / "this_kernel_doesnt_exist.py",
            arguments_order=["id"],
        )

        with pytest.raises(FileNotFoundError):
            _ = runner({"id": 5})


    def test_invalid_permissions(self, tmp_path: pathlib.Path):
        
        # windows does not have a simple execution permission file attribute
        if os.name == "nt":
            return

        shutil.copy(pathlib.Path(__file__).parent / "dummy_kernel.py", tmp_path / "dummy_kernel.py")
        # Remove execution permission
        os.chmod(tmp_path / "dummy_kernel.py", 0o666)

        runner = MonoSubprocessHarness(
            objectives=["r"],
            objectives_bounds={"r": 42},
            executable_path=tmp_path / "dummy_kernel.py",
            arguments_order=["id"],
        )

        with pytest.raises(PermissionError):
            _ = runner({"id": 5})

        os.chmod(tmp_path / "dummy_kernel.py", 0o000)

        with pytest.raises(PermissionError):
            _ = runner({"id": 5})

        

