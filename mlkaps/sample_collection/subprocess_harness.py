"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

from dataclasses import dataclass
import subprocess
import textwrap
from contextlib import suppress
import os
import signal
import logging
import pathlib
from collections import namedtuple


@dataclass
class ProcessResult:
    stdout: str
    stderr: str
    exitcode: int
    timed_out: bool
    arguments: list[str]
    pid: int


class ProcessCleanupHandler:
    """
    Helper class to cleanly kill all subprocess on failure or user exit

    This class wraps around subprocess.Popen and ensures that the subprocess and its childs are killed if an exception if received.
    This includes SystemExit and KeyboardInterrupt, so no zombie process is created if the user kills MLKAPS.
    """

    def __init__(self, timeout: float = None):
        """Initiliaze the cleanup handler

        :param timeout: A timeout in seconds for the process to finish. The process will be killed if the timeout expries, defaults to None
        :type timeout: float, optional
        """
        self.timeout = timeout

    def _cleanup_kill(self, process: subprocess.Popen) -> tuple[str, str]:
        """Forcefully kills a process (if alive) and returns its stdout/stderr output

        :param process: The handler returned by Popen for the process
        :type process: subprocess.Popen
        :return: The stdout and stderr of the process
        :rtype: tuple[str, str]
        """

        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return process.communicate()

    def run(self, *args, **kwargs) -> ProcessResult:
        """Forwards all arguments to subprocess.Popen and ensures the process if an exception is raised

        :return: The process handle returned by Popen, the stdout and stderr of the process
        :rtype: tuple[subprocess.Popen, str, str]
        """

        timed_out = False
        process = None
        try:
            process = subprocess.Popen(*args, **kwargs)
            stdout, stderror = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            logging.warning(f"Process timedout, cleaning up...)")

            stdout, stderror = self._cleanup_kill(process)
            timed_out = True
        except:
            # Can occur if an exception is raised  during Popen
            if process is None:
                raise

            logging.warning(
                f"Exception raised during subprocess execution, cleaning up..."
            )

            self._cleanup_kill(process)
            raise

        return ProcessResult(
            stdout=stdout,
            stderr=stderror,
            exitcode=process.returncode,
            timed_out=timed_out,
            arguments=process.args,
            pid=process.pid,
        )


class MonoSubprocessHarness:

    def __init__(
        self,
        objectives: list[str],
        executable_path: pathlib.Path,
        arguments_order: list[str],
        timeout: float | None = None,
    ):
        self.objectives = objectives
        self.executable_path = executable_path
        self.timeout = timeout
        self.arguments_order = arguments_order

    def _parse_output(self, stdout: str) -> dict:
        last_line = stdout.strip().split("\n")[-1]

        objectives = last_line.split(",")

        try:
            objectives = [float(o) for o in objectives]
        except ValueError:
            return None, f"Invalid output: {last_line}"

        if len(objectives) != len(self.objectives):
            return (
                None,
                f"Invalid number of outputs, expected {len(self.objectives)}, got {len(objectives)}",
            )

        return {o: v for o, v in zip(self.objectives, objectives)}, None

    def _run_process(self, arguments: list[str]) -> ProcessResult:
        arguments = [self.executable_path, *arguments]

        handler = ProcessCleanupHandler(self.timeout)
        result = handler.run(
            arguments,
            text=True,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        return result

    def __call__(self, sample: dict):

        arguments = [str(sample[k]) for k in self.arguments_order]
        result = self._run_process(arguments)

        if result.timed_out:
            error = f"Process timed out (max {self.timeout} seconds)\n"
        elif result.exitcode != 0:
            error = f"Process exited with code {result.exitcode}\n"
        else:
            objectives, error = self._parse_output(result.stdout)

        if error is not None:
            objectives = {o: float("nan") for o in self.objectives}
            msg = textwrap.indent(f"Arguments: {result.arguments}", "\t| ")
            msg += textwrap.indent(f"\nPID: {result.pid}", "\t| ")
            msg += textwrap.indent(f"\nStdout:\n{result.stdout}", "\t> ")
            error += msg

        res_type = namedtuple("SubprocessRunnerOutput", ["data", "error", "timed_out"])
        return res_type(data=objectives, error=error, timed_out=result.timed_out)
