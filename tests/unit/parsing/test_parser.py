"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import json
import os
import pathlib
import unittest
import pytest

from mlkaps.configuration import ExperimentConfig, _parser


def _fetch_dummy_json(json_name):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "dummy_jsons", json_name)
    if not os.path.exists(path):
        print("File not found: {}".format(path))
    res = json.load(open(path))
    return res, path


class BasicParserTestCase(unittest.TestCase):

    def test_throw_on_missing_sections(self):
        res, path = _fetch_dummy_json("missing_sections.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)

    def test_throw_on_empty_json(self):
        res, path = _fetch_dummy_json("empty_json.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)

    def test_throw_on_invalid_path(self):
        self.assertRaises(Exception, ExperimentConfig.from_dict, None)

    def test_can_parse_valid_json(self):
        res, path = _fetch_dummy_json("valid_json.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        config = ExperimentConfig.from_dict(res, pathlib.Path(path).parent)
        self.assertIsNotNone(config)


class TestInvalidConfig:
    @pytest.mark.parametrize(
        "configfile",
        [
            "parameters/missing_design_parameters.json",
            "parameters/missing_kernel_inputs.json",
            "parameters/unknown_parameter_type.json",
            "parameters/invalid_numerical_parameter_value.json",
        ],
    )
    @pytest.mark.xfail(raises=KeyError, strict=True)
    def test_keys(self, configfile):
        res, path = _fetch_dummy_json(configfile)
        assert res, "Missing dummy json file"
        ExperimentConfig.from_dict(res, pathlib.Path(path).parent)

    @pytest.mark.parametrize(
        "configfile",
        [
            "parameters/conflicting_containers.json",
            "parameters/no_container.json",
            "parameters/invalid_container.json",
            "parameters/invalid_sequence_type.json",
            "parameters/invalid_range_type.json",
            "parameters/invalid_progression.json",
            "parameters/invalid_sequence.json",
            "parameters/invalid_range.json",
        ],
    )
    @pytest.mark.xfail(raises=_parser.ParserError, strict=True)
    def test_invalid_parameter(self, configfile):
        res, path = _fetch_dummy_json(configfile)
        assert res, "Missing dummy json file"
        ExperimentConfig.from_dict(res, pathlib.Path(path).parent)


class SamplingTestCase(unittest.TestCase):

    def test_throw_on_invalid_scripts(self):
        res = _fetch_dummy_json("sampling/invalid_scripts.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)

    def test_throw_on_missing_executable(self):
        # The user MAY define an empty executable path (we'll just crash if it is called),
        # but anyhow the executable field is MUST be defined
        res = _fetch_dummy_json("sampling/missing_executable.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)

    def test_throw_on_missing_objectives(self):
        # The user MAY define an empty executable path (we'll just crash if it is called),
        # but anyhow the executable field is MUST be defined
        res = _fetch_dummy_json("sampling/missing_objectives.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)

    def test_throw_on_unknown_sampling_method(self):
        # The user MAY define an empty executable path (we'll just crash if it is called),
        # but anyhow the executable field is MUST be defined
        res = _fetch_dummy_json("sampling/unknown_sampling_method.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)


if __name__ == "__main__":
    unittest.main()
