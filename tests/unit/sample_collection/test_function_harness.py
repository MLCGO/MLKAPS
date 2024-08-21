"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.sample_collection.function_harness import MonoFunctionHarness
from mlkaps.sample_collection.function_harness import FunctionPath
import time
import math
import pathlib
import pytest


class TestFunctionPath:

    def test_can_load_simple(self):
        path = FunctionPath("math.sqrt")

        assert path.stem == "sqrt"
        assert path.parents == "math"
        assert path.path == "math.sqrt"
        assert (func := path.to_function()) == math.sqrt
        assert func(9) == 3.0

    def test_can_load_from_file(self):

        path = FunctionPath(pathlib.Path(__file__).parent / "dummy_module.py:my_sqrt")

        assert path.stem == "my_sqrt"
        assert path.parents == str(
            pathlib.Path(__file__).parent / "dummy_module.py"
        )
        assert path.path == str(
            pathlib.Path(__file__).parent / "dummy_module.py:my_sqrt"
        )
        assert path.to_function()({"id": 9})["r"] == 3.0

    def test_can_change_function_name(self):

        path = FunctionPath("math.sqrt")
        _ = path.to_function()

        path.stem = "exp"

        assert path.stem == "exp"
        assert path.parents == "math"
        assert path.path == "math.exp"
        assert path.to_function()(2) == math.exp(2)

    def test_can_change_function_name_from_file(self):

        path = FunctionPath(pathlib.Path(__file__).parent / "dummy_module.py:my_sqrt")
        _ = path.to_function()

        path.stem = "return_id"

        assert path.stem == "return_id"
        assert path.parents == str(
            pathlib.Path(__file__).parent / "dummy_module.py"
        )
        assert path.path == str(
            pathlib.Path(__file__).parent / "dummy_module.py:return_id"
        )
        assert path.to_function()({"id": 2})["r"] == 2

    def test_raises_on_invalid_path(self):

        with pytest.raises(AttributeError):
            FunctionPath("math.no_such_function").to_function()

        with pytest.raises(ValueError):
            FunctionPath("math:no_such_module").to_function()

        with pytest.raises(AttributeError):
            FunctionPath(
                pathlib.Path(__file__).parent / "dummy_module.py:no_such_function"
            ).to_function()

        with pytest.raises(ValueError):
            FunctionPath(
                pathlib.Path(__file__).parent / "dummy_module.py:my_sqrt:here"
            ).to_function()


    def test_detect_type(self):
        path = FunctionPath("..my_module.math")
        assert path.is_module() and path.is_relative() and not path.is_source()

        path = FunctionPath(".my_module..my_module2.math")
        assert path.is_module() and path.is_relative() and not path.is_source()

        path = FunctionPath("....my_module..math")
        assert path.is_module() and path.is_relative() and not path.is_source()

        path = FunctionPath("my_module.math")
        assert path.is_module() and not path.is_relative() and not path.is_source()

        path = FunctionPath("my_module.py:math")
        assert path.is_source() and path.is_relative() and not path.is_module()

        path = FunctionPath("../../my_module.py:math")
        assert path.is_source() and path.is_relative() and not path.is_module()

        path = FunctionPath(pathlib.Path("my_module.py:math").resolve())
        assert path.is_source() and not path.is_relative() and not path.is_module()

        assert FunctionPath("math.sqrt").is_module() and FunctionPath("math.sqrt").exists()

    def test_detect_type_on_invalid_path(self):
        path = FunctionPath("$$$$$.:math:invalid")
        assert not path.is_module() and not path.is_source() and not path.exists()


class TestMonoFunctionRunner:

    def test_can_sample(self):
        runner = MonoFunctionHarness(lambda x: {"r": x["id"]}, ["r"])
        samples = [{"id": 1}, {"id": 2}, {"id": 3}]

        for s in samples:
            result = runner(s)
            assert result.data == {"r": s["id"]}
            assert result.error is None
            assert not result.timed_out

    def test_detects_invalid_return(self):
        # Check we can detect invalid number of return when using dict
        def test_all_none(runner):
            samples = [{"id": 1}, {"id": 2}, {"id": 3}]
            for s in samples:
                result = runner(s)
                # we should report an error on all cases since the kernel won't report the correct number of arguments
                assert result.error is not None

        lambdas = [
            lambda _: {},
            lambda _: None,
            lambda _: 42,
            lambda x: [x["id"], -x["id"]],
        ]

        for l in lambdas:
            runner = MonoFunctionHarness(l, ["r"])
            test_all_none(runner)

    def test_can_timeout(self):

        def functor(sample):
            # Take five second if id is above 5
            if sample["id"] < 5:
                time.sleep(5)
            return {"r": sample["id"]}

        runner = MonoFunctionHarness(functor, ["r"], timeout=0.1)
        samples = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 6}, {"id": 7}]

        for s in samples:

            begin = time.time()
            result = runner(s)
            end = time.time()

            # The runner should timeout if the id is above 5
            if s["id"] < 5:
                assert result.error is not None
                assert result.timed_out
                assert end - begin < 0.2
            else:
                # Otherwise, the runner should return the result
                assert result.data == {"r": s["id"]}
                assert result.error is None
                assert not result.timed_out
