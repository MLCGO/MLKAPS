"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os
import unittest
import json
import pathlib
from mlkaps.configuration import ExperimentConfig


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


class ParametersTestCase(unittest.TestCase):

    def test_throw_on_missing_design_parameters(self):
        res, path = _fetch_dummy_json("parameters/missing_design_parameters.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)

    def test_throw_on_missing_kernel_inputs(self):
        res, path = _fetch_dummy_json("parameters/missing_kernel_inputs.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)

    def test_throw_on_unknown_parameter_type(self):
        res, path = _fetch_dummy_json("parameters/unknown_parameter_type.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)

    def test_throw_on_invalid_numerical_parameter_value(self):
        res, path = _fetch_dummy_json(
            "parameters/invalid_numerical_parameter_value.json"
        )
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res, path)


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
