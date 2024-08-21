"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
from typing import Callable
import importlib
from collections import namedtuple
import multiprocessing
from .common import KernelSamplingError
import re


_source_pattern = re.compile(r"^(?P<path>.+\.py):(?P<function>\S+)$")
_module_pattern = re.compile(r"^(?P<path>(?:\w|\.)+)\.(?P<function>[^\s:]+)$")


class FunctionPath:
    """
    Class to represent a path to a python function.

    Can represent either a module path (i.e math.sqrt) or a source path (i.e /path/to/module.py:function_name).
    Provide facilities to import the function from the path (with caching).
    """

    def __init__(self, function_path: str | pathlib.Path):
        """
        Create a new FunctionPath object

        :param function_path: Path to the function, can be either a module path or a source path
        :type function_path: str | pathlib.Path
        """

        self.path = function_path
        # Cache system for the function
        self._functor = None

    def _make_path(self, path: str | pathlib.Path) -> str:
        # duck typing for pathlib.Path
        if (resolve := getattr(path, "resolve", None)) is not None and callable(
            resolve
        ):
            return str(resolve())
        return path

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str | pathlib.Path):
        self._path = self._make_path(value)

        # Invalidate the cache
        self._functor = None

    @property
    def stem(self) -> str:
        """
        Retrieve the function name from the path, i.e math.sqrt -> sqrt, my_module.py:sqrt -> sqrt

        :return: The name of the function
        :rtype: str
        """
        if self.path == "" or self.path is None:
            return ""
        elif self.is_source():
            return self.path.split(":")[-1]
        else:
            return self.path.rsplit(".", 1)[-1]

    @stem.setter
    def stem(self, value: str):
        """
        Modify the function name for the path, i.e math.sqrt -> math.value, my_module.py:sqrt -> my_module.py:value

        :param value: The new name for the function
        :type value: str
        """
        if self.is_source():
            self.path = f"{self.parents}:{value}"
        else:
            self.path = f"{self.parents}.{value}"

    @property
    def parents(self) -> str:
        """
        Return the parents from the path, i.e my_module.math.sqrt -> my_module.math, my_module.py:sqrt -> my_module

        :return: The name of the module
        :rtype: str
        """
        if self.is_source():
            return self.path.split(":")[0]
        else:
            return self.path.rsplit(".", 1)[0]

    @parents.setter
    def parents(self, value: str):
        """
        Modify the parents for the path, i.e my_module.math.sqrt -> value.sqrt, my_module.py:sqrt -> value.py:sqrt

        :param value: The new parents for the function
        :type value: str
        """
        if self.is_source():
            self.path = f"{value}:{self.stem}"
        else:
            self.path = f"{value}.{self.stem}"

    def exists(self) -> bool:
        """
        :return: Return True if this path exists and is importable
        :rtype: bool
        """
        res = self.to_function(none_on_failure=True)
        return not res is None

    def _import_from_source(self) -> Callable:
        """
        Import the function from a file on disk

        :raises FileNotFoundError: Raised if the file does not exist
        :return: A functor to the function
        :rtype: Callable
        """

        module_path, function_name = self.path.split(":")

        if not pathlib.Path(module_path).exists():
            raise FileNotFoundError(f"Module {module_path} does not exist")

        spec = importlib.util.spec_from_file_location(module_path, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, function_name)

    def _import_from_module(self) -> Callable:
        """
        Import the function from a module in the python path

        :return: A functor to the function
        :rtype: Callable
        """

        module_path, function_name = self.path.rsplit(".", 1)
        module = importlib.import_module(module_path)

        return getattr(module, function_name)

    def to_function(self, none_on_failure=False):
        """
        Attempts to import the function from the path.

        Will return default on failure, or raise if default is None or not provided.

        :return: A callable if the function is found, default otherwise
        :rtype: Callable | None
        """

        # Cache the import result
        if self._functor is not None:
            return self._functor

        print(self.path, self.is_source(), self.is_module())
        try:
            if self.is_source():
                res = self._import_from_source()
            elif self.is_module():
                res = self._import_from_module()
            elif none_on_failure:
                return None
            else:
                raise ValueError(f"Invalid path {self.path}")
        except (ModuleNotFoundError, FileNotFoundError):
            if none_on_failure is None:
                raise
            return None

        self._functor = res
        return res

    def is_source(self) -> bool:
        """
        :return: Return true if the path corresponds to a python source file
        :rtype: bool
        """

        return _source_pattern.match(self.path) is not None

    def is_module(self) -> bool:
        """
        :return: Return true if the path corresponds to a python module
        :rtype: bool
        """
        return _module_pattern.match(self.path) is not None

    def is_relative(self) -> bool:
        """
        For source paths, this means the path is not absolute.
        For module paths, this means the path contains a relative import.

        :return: Return true if this path is relative
        :rtype: bool
        """
        if self.is_source():
            path = pathlib.Path(self.path.split(":")[0])
            return not path.is_absolute()

        return ".." in self.path or self.path.startswith(".")

    def __repr__(self):
        return (
            f"FunctionPath({self.path=}" + f", {self._functor=})"
            if self._functor is not None
            else ")"
        )

    def __str__(self):
        return f"[Function Path] <{self.path}>"


class FunctionTimeoutWrapper:

    def __init__(self, function: Callable[[dict], dict], timeout: float | None = None):

        if not callable(function):
            raise ValueError(f"Function must be a callable object, received {function}")

        self.function = function
        self.timeout = timeout

    def _start_subprocess(self, args, kwargs):
        # Ensure that the process with either push the result to the queue, or an exception
        def wrapper(queue, args, kwargs):
            try:
                queue.put(self.function(*args, **kwargs))
            except Exception as e:
                queue.put(e)

        # Create a queue for the subprocess to push to
        queue = multiprocessing.Queue()

        # Start the subprocess and wait for timeout or finish
        proc = multiprocessing.Process(
            target=wrapper,
            args=(
                queue,
                args,
                kwargs,
            ),
        )
        proc.start()
        return proc, queue

    def _join_subprocess(self, proc, queue):
        proc.join(self.timeout)

        # If exitcode is None, this means the function is still running and we must time out
        if proc.exitcode is None:
            proc.terminate()
            raise TimeoutError(
                f"Function timed out after {self.timeout} seconds\n{self=}"
            )

        # Else, we can get the output from the queue
        output = queue.get()

        # Re-raise any caught exception
        if isinstance(output, Exception):
            raise KernelSamplingError(f"Function execution failed\n{self=}") from output

        return output

    def _call_in_subprocess(self, args, kwargs):
        proc = None
        try:
            proc, queue = self._start_subprocess(args, kwargs)
            output = self._join_subprocess(proc, queue)
        except:
            # Clean up the process if it is still running, even if we caught a KeyboardInterrupt or similar
            # base exception
            if proc is not None and proc.is_alive():
                proc.terminate()
            raise

        return output

    def __call__(self, *args, **kwargs):
        if self.timeout is not None:
            return self._call_in_subprocess(args, kwargs)

        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            raise KernelSamplingError(f"Function execution failed\n{self=}") from e

    def __repr__(self):
        return f"FunctionTimeoutWrapper({self.function=}, {self.timeout=})"

    def __str__(self):
        return f"[Function Timeout Wrapper] <{self.function}>"


class MonoFunctionHarness:

    def __init__(
        self,
        function: Callable[[dict], dict] | str | pathlib.Path | FunctionPath,
        expected_keys: list[str],
        timeout: float | None = None,
    ):
        if isinstance(function, (pathlib.Path, str)):
            function = FunctionPath(function).to_function()

        if (to_function := getattr(function, "to_function", None)) is not None:
            function = to_function()

        if not callable(function):
            raise ValueError(f"Function must be a callable object, received {function}")

        self.function = function
        self.timeout = timeout
        self.expected_keys = expected_keys

    def _verify_result(self, result):

        if not isinstance(result, dict):
            raise KernelSamplingError(
                f"Expected a dict as output, received {type(result)} ({result})"
            )

        if not all(k in result for k in self.expected_keys):
            raise KernelSamplingError(
                f"Expected keys {self.expected_keys} not found in result {result}"
            )

    def __call__(self, sample: dict):
        res_type = namedtuple("FunctionRunnerOutput", ["data", "error", "timed_out"])
        try:
            result = FunctionTimeoutWrapper(self.function, self.timeout)(sample)

            self._verify_result(result)

            # Merge the sample and the output into a single dict
            return res_type(data=result, error=None, timed_out=False)
        except Exception as e:
            return res_type(
                data={k: float("nan") for k in self.expected_keys},
                error=str(e),
                timed_out=isinstance(e, TimeoutError),
            )
