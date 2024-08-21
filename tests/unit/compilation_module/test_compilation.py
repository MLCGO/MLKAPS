"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import json
import os
import unittest

from mlkaps.configuration.compilation_configuration import register_predefined_flag_set
from mlkaps.configuration import ExperimentConfig


def _fetch_dummy_json(json_name):
    base = os.path.dirname(os.path.abspath(__file__)) + "/dummy_jsons"
    path = os.path.join(base, json_name)
    if not os.path.exists(path):
        print("File not found: {}".format(path))
    res = json.load(open(path))
    return res, base


class ParsingCompilationTestCase(unittest.TestCase):

    def test_produce_configuration(self):
        res, path = _fetch_dummy_json("valid_json.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        config = ExperimentConfig.from_dict(res, path)
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.compilation_configuration)

    def test_raise_on_invalid_compilation_script(self):
        res = _fetch_dummy_json("missing_compilation_script.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)

        res = _fetch_dummy_json("non_executable_compilation_script.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        self.assertRaises(Exception, ExperimentConfig.from_dict, res)

    def test_can_parse_static_flags(self):
        res, path = _fetch_dummy_json("static_flags.json")
        if res is None:
            self.skipTest("Missing dummy json file")

        config = ExperimentConfig.from_dict(res, path)
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.compilation_configuration)
        self.assertEqual(
            config.compilation_configuration.get_static_flags(), ["-O3", "-g"]
        )

    def test_can_parse_predefined_sets(self):
        register_predefined_flag_set(
            "test_set_please_ignore",
            {
                "debug_1": {"command_line": "-DDEBUG_1", "Type": "Boolean"},
                "debug_2": {"command_line": "-DDEBUG_2", "Type": "Boolean"},
            },
        )
        res, path = _fetch_dummy_json("predefined_sets.json")
        if res is None:
            self.skipTest("Missing dummy json file")
        print(res)

        config = ExperimentConfig.from_dict(res, path)
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.compilation_configuration)
        self.assertEqual(
            list(config.compilation_configuration.flag_features.keys()),
            ["debug_1", "debug_2"],
        )


if __name__ == "__main__":
    unittest.main()
