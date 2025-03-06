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
import os
import pickle


def is_pickleable(obj):
    try:
        pickle.dumps(obj)
    except Exception as e:
        return False
    return True
    

_source_pattern = re.compile(r"^(?P<path>.+\.py):(?P<function>\S+)$")
_module_pattern = re.compile(r"^(?P<path>(?:\w|\.)+)\.(?P<function>[^\s:]+)$")




class Functor:
    """
    On Windows (os.name="nt"):
    
        If we have a timeout, we create a subprocess.  All the objects that are passe 
        the subprocess are pickled. But there are numerous limitations on what functions 
        can get pickled, and in particular, functions that are imported from a source file
        with importlib cannot be pickled.  For these functions, we need to pass the file and 
        function name to the subprocess, and re-import the function in the subprocess.  

        If we do not have a timeout, we do not create a subprocess.  In this case, we can
        just call the function directly.

    On Linux:
        On Linux, we fork if have a timeout, and all the objects exist in the subprocess.  
        Nothing needs to pickled.  We only need the function.

    """
    def __init__(self, function: Callable,  callable_in_subprocess: bool = True,  module_path: str | None = None, function_name: str | None = None):

        self._internal_fn = function

        # these are needed for Windows 
        self.is_callable_in_subprocess = callable_in_subprocess 
        self._module_path = module_path
        self._function_name = function_name
      
    def is_callable(self):
        # On Windows, this means there is path to calling the function, but we may
        # hit a problem later if we try to call it in a subprocess
        if os.name == "nt":
            if callable(self._internal_fn):  # and is_pickleable(self._internal_fn):
                return True
            elif self._module_path is not None and self._function_name is not None:
                return True 
            else:
                return False
        else: 
            return callable(self._internal_fn) 
        
    def get_callable_function(self):
        if os.name == "nt" and not self.is_callable_in_subprocess:
            # On Windows, loading this function here may cause problems if we try to call it in a subprocess
            spec = importlib.util.spec_from_file_location(self._module_path, self._module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, self._function_name)
        elif self._internal_fn is not None:
            return self._internal_fn
        else:
            raise ValueError("No function to return")   



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
            split_name = self.path.split(":")
            if os.name == "nt" and len(split_name) == 3:
                 # on Windows, the self.path will look like C:\path\to\file.py:function_name
                drive, file_path, function_name = split_name
                return f"{drive}:{file_path}"
            elif len(split_name) == 2:
                file_path, function_name = split_name
                return file_path
            else:
                raise ValueError(f"Invalid path {self.path}") 
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
        :raises Exception: if we cannot understand the colons in the path
        :return: A functor to the function
        :rtype: Callable
        """
        split_name = self.path.split(":")
        if os.name == "nt" and len(split_name) == 3:
                 # on Windows, the self.path will look like C:\path\to\file.py:function_name
            drive, file_path, function_name = split_name
            module_path = f"{drive}:{file_path}"
        elif len(split_name) == 2:
            module_path, function_name = split_name
        else:
            raise ValueError(f"Invalid path {self.path}")

        if not pathlib.Path(module_path).exists():
            raise FileNotFoundError(f"Module {module_path} does not exist")

        if os.name == "nt":
            # On Windows, if we load the function now with spec_from_file_location, we will
            # have a problem if we want to call it in a subprocess.
            return(Functor(None, callable_in_subprocess=False, module_path=module_path, function_name=function_name))
        else:
            try:
                spec = importlib.util.spec_from_file_location(module_path, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                fn = getattr(module, function_name) 
            except Exception:
                raise   
            return(Functor(fn))   


    def _import_from_module(self) -> Callable:
        """
        Import the function from a module in the python path

        :return: A functor to the function
        :rtype: Callable
        """

        module_path, function_name = self.path.rsplit(".", 1)
        
        try:
            module = importlib.import_module(module_path)
            functor = getattr(module, function_name) 
        except Exception:
            raise
          
        return(Functor(functor))


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
        if os.name == "nt":
             # on Windows, the self.path will look like C:\path\to\file.py:function_name
             # if this path begins with a drive letter, split it off
            path_list = self.path.split(":",1)
            if len(path_list) == 2 and len(path_list[0]) == 1 and path_list[0].isalpha():
                return _module_pattern.match(path_list[1]) is not None
            
        return _module_pattern.match(self.path) is not None

    def is_relative(self) -> bool:
        """
        For source paths, this means the path is not absolute.
        For module paths, this means the path contains a relative import.

        :return: Return true if this path is relative
        :rtype: bool
        """
        if self.is_source():
            split_name = self.path.split(":")
            if os.name == "nt" and len(split_name) == 3:
                 # on Windows, the self.path will look like C:\path\to\file.py:function_name
                drive, file_path, function_name = split_name
                path = f"{drive}:{file_path}"
                return not pathlib.Path(path).is_absolute()
            elif len(split_name) == 2:
                path, function_name = split_name
                return not pathlib.Path(path).is_absolute()
            else:
                raise ValueError(f"Invalid path {self.path}") 
                      
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

        if not function.is_callable():
            raise ValueError(f"Function must be a callable object, received {function}")

        self.function = function
        self.timeout = timeout

    # On Windows, the wrapper function must be pickleable, so definition needs to be at top-level of class
    # This function ensures that the process will either push the result to the queue, or raise an exception
    def wrapper(self,queue, args, kwargs):
        try:
            functor = self.function.get_callable_function()
            queue.put(functor(*args, **kwargs))
        except Exception as e:
            queue.put(e)

    # On Windows, we will import the file in the subprocess.  It is important that we
    # do not import the file in the parent process.  If we do import the file in the parent,
    # we will not be able to pickle the functions.
    def wrapper_from_file_location(self,queue, module_path, function_name, args, kwargs):
        try:
            spec = importlib.util.spec_from_file_location(module_path, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            functor = getattr(module, function_name)
            queue.put(functor(*args, **kwargs))
        except Exception as e:
            queue.put(e)



    def _start_subprocess(self, args, kwargs):
        # Create a queue for the subprocess to push to
        queue = multiprocessing.Queue()

        # Start the subprocess and wait for timeout or finish 
        if os.name == "nt" and not self.function.is_callable_in_subprocess:
            assert is_pickleable(self.wrapper_from_file_location), "wrapper not pickleable"
            module_path = self.function._module_path
            function_name = self.function._function_name
            proc = multiprocessing.Process(
                target=self.wrapper_from_file_location,
                args=(queue, module_path, function_name, args, kwargs,),
            )
        else:
            proc = multiprocessing.Process(
                target=self.wrapper,
                args=(queue, args, kwargs, ),
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
        except Exception as e:
            print("Exception in call_in_subprocess:",e)
            print(self, args, kwargs)
            # Clean up the process if it is still running, even if we caught a KeyboardInterrupt or similar
            # base exception
            if proc is not None and proc.is_alive():
                proc.terminate()
            raise

        return output
 
    def _call_in_current_process(self, *args, **kwargs):
        try:
            return self.function.get_callable_function()(*args, **kwargs)
        except Exception as e:
            raise KernelSamplingError(f"Function execution failed\n{self=}") from e 



    def __call__(self, *args, **kwargs):
        if self.timeout is not None:
            return self._call_in_subprocess(args, kwargs)
        else:
            return self._call_in_current_process(*args, **kwargs)
    

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
        if not callable(function):
            if isinstance(function, (pathlib.Path, str)):
                function = FunctionPath(function).to_function()

            if (to_function := getattr(function, "to_function", None)) is not None:
                function = to_function()

            if not function.is_callable():
                raise ValueError(f"Function must be a callable object, received {function}")
        else:
            function = Functor(function)

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
