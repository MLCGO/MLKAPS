"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os

from ._parsing_helpers import assert_file_is_executable

# Pre-defined sets of flags
# Defined as:
#   "set_name": {
#    "flag_1_name": {
#       "command_line": "flag_1_command_line",
#       "Type": "Boolean"
#    },
#    "flag_2_name": {
#       ...
flag_sets_list = {"ffast-math": {"-ffast-math": {"command_line": "-ffast-math", "Type": "Boolean"}}}


def register_predefined_flag_set(set_name: str, flag_set: dict):
    """
    Add a new set of flags to the list of pre-defined sets

    Parameters
    ----------
    set_name : str
        The name of the flag set to be defined
    flag_set : dict
        The dictionary containing the flag set to be defined
    """
    flag_sets_list[set_name] = flag_set


def _check_has_compilation_flags(config_dict):
    """
    Check if the json file contains a "COMPILATION" section
    """
    return "COMPILATION" in config_dict["PARAMETERS"]


class CompilationConfiguration:
    """
    Configuration object to hold all settings related to the compilation module and the
    compilation of the kernel itself

    Note that this module is still a prototype, and that the current implementation
    could be subject to change in the future

    Currently, we only support static and predefined flags
    """

    @staticmethod
    def make_if_enabled(parent_configuration, configuration_dict):
        """
        Build a CompilationConfiguration object if the json file contains a "COMPILATION"
        section, else return None
        """
        if _check_has_compilation_flags(configuration_dict):
            return CompilationConfiguration(parent_configuration, configuration_dict)
        else:
            return None

    def __init__(self, parent_configuration, configuration_dict):
        """
        Initialize a new CompilationConfiguration object from the passed json dict
        """
        self.keys = {}

        if not _check_has_compilation_flags(configuration_dict):
            self.is_valid = False
            return

        self.is_valid = True

        self.parent = parent_configuration
        self.flag_features = {}
        # Fetch user defined flags
        self._fetch_flags(configuration_dict)
        self._fetch_compilation_scripts(configuration_dict)

    def _fetch_static_flags(self, configuration_dict):
        """
        Fetch flags that should always be used for compilation, and are not design features
        """
        self.static_flags = []

        if "static_flags" in configuration_dict["PARAMETERS"]["COMPILATION"]:
            self.static_flags = configuration_dict["PARAMETERS"]["COMPILATION"]["static_flags"]

    def _unfold_flag_set(self, set_name):
        """
        Unfold a pre-defined flag set into one or more flag design features
        """
        if set_name not in flag_sets_list:
            raise Exception("CompilationConfiguration - Unknown flag set {}".format(set_name))

        for k, v in flag_sets_list[set_name].items():
            self.flag_features[k] = v

    def _fetch_predefined_flag_set(self, configuration_dict):
        """
        Fetch and unfold every pre-defined flag set inside the configuration dict
        """
        predefined_sets = configuration_dict["PARAMETERS"]["COMPILATION"]["predefined_flag_sets"]

        for s in predefined_sets:
            self._unfold_flag_set(s)

    def _make_design_feature_from_flag(self):
        """
        Turn a user-defined flag into a design feature for the experiment
        """
        self.parent.add_features(self.flag_features)
        self.parent.promote_features_to_design(self.flag_features.keys())

    def _fetch_flags(self, json):
        """
        Check for the compilation flags to add as design features
        """
        self._fetch_static_flags(json)
        self._fetch_predefined_flag_set(json)

        # custom flags are not yet supported
        # self.__fetch_custom_flags(json)
        self._make_design_feature_from_flag()

    def _fetch_compilation_scripts(self, json_file):
        """
        Fetch the compilation script from the json file, and ensure
        it is executable
        """
        compilation_section = json_file["PARAMETERS"]["COMPILATION"]

        if "compilation_script" not in compilation_section:
            raise Exception("Compilation script is missing from the json input file")

        self.compilation_script = compilation_section["compilation_script"]

        # Ensure the path is absolute
        if not os.path.isabs(self.compilation_script):
            self.compilation_script = os.path.join(self.parent.get_working_directory(), self.compilation_script)

        assert_file_is_executable(self.compilation_script)

    def get_static_flags(self):
        return self.static_flags

    def get_compilation_script(self):
        return self.compilation_script

    def get_static_flags_as_string(self):
        return " ".join(self.get_static_flags())
