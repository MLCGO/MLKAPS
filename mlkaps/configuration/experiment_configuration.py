"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import os
import pathlib

from deprecated import deprecated
from . import _parser as parser
from mlkaps.configuration.compilation_configuration import (
    CompilationConfiguration,
)


class ExperimentConfig:
    """
    Configuration class that hold all the information needed to run an experiment, as well as most
    parameters for every MLKAPS module.

    This object if designed to be built from a json/dictionary containing the configuration of the
    experiment.

    """

    @staticmethod
    def from_dict(
        config_dict: dict, working_directory: str = None, output_directory: str = None
    ):
        """
        Build a ExperimentConfig object from a dictionary containing the configuration of the
        experiment
        """

        # if the config path is not defined, default to the current working directory
        if not working_directory:
            working_directory = os.getcwd()

        abs_wd_path = pathlib.Path(working_directory).absolute()
        return ExperimentConfig(config_dict, abs_wd_path, output_directory)

    def __init__(
        self,
        config_dict,
        working_directory: pathlib.Path,
        output_directory: pathlib.Path | None = None,
    ):
        """
        Builds a new experiment configuration, fetching data from the passed configuration
        dictionary. Prefer using the factory method from_dict()

        Parameters
        ----------
        config_dict : dict
            The dictionary containing the configuration of the experiment and MLKAPS modules
        working_directory : str
            The path to the root directory of the experiment, where the kernel/execution script
            is located
        output_directory : str
            The path to the directory where MLKAPS output will be stored
            If undefined, defaults to "{working_directory}/mlkaps_output"
        """

        self._build_work_directories(config_dict, working_directory, output_directory)

        # Extract every needed section from the json file
        self.keys = {}
        parser.parse_experiment_parameters(self, config_dict)

        # Optionally parse the optimization flags
        self.compilation_configuration = CompilationConfiguration.make_if_enabled(
            self, config_dict
        )

        parser.parse_objectives(self, config_dict)
        self.objectives = self["experiment"]["objectives"]

        parser.parse_modeling(self, config_dict)
        parser.parse_clustering(self, config_dict)

        self.design_parameters = self["parameters"]["optimization_features"]
        self.input_parameters = self["parameters"]["inputs"]
        self.feature_values = self["parameters"]["features_values"]
        self.parameters_type = self["parameters"]["features_type"]

    def _build_work_directories(
        self,
        config_dict: dict,
        working_directory: pathlib.Path,
        output_directory: pathlib.Path,
    ):
        if not working_directory.exists():
            raise parser.ParserError(
                f"The working directory '{working_directory}' does not exists, exiting"
            )

        self.working_directory = working_directory

        if output_directory is not None:
            output_directory = pathlib.Path(output_directory)
        else:
            output_directory = pathlib.Path(
                config_dict["EXPERIMENT"].get(
                    "output_directory", working_directory / "mlkaps_output/"
                )
            )

        self.output_directory = output_directory.absolute()

        if not self.output_directory.exists():
            os.makedirs(self.output_directory)

        self.metadata_directory = self.output_directory / "metadata"

    def get_working_directory(self):
        return self.working_directory

    def add_features(self, parameters_to_promote: dict):
        """
        This function takes a dictionary of parameters (A tuple of name, type and values) and
        promote them to features.

        The parameters should be passed as a dictionary, where the key is the name of the parameter
        containing the "Type" subkey, one of "Categorical, "Boolean", "Integer" or "Float",
        and the "Values" subkey, a list of values for the parameter if it is categorical, or a
        tuple of (min, max) values if it is an integer or a float.

        Parameters
        ----------
        parameters_to_promote : dict
            A dictionary of parameters to promote to features.
        """
        keys = self["parameters"]

        # Parse every entry in the dictionary
        # And append their valuer/type to the feature map
        for p, v in parameters_to_promote.items():
            keys["features_type"][p] = v["Type"]
            # If the parameter is boolean, deduce the values to False/True
            if keys["features_type"][p] == "Boolean":
                keys["features_values"][p] = [False, True]
            else:
                # Else, just copy the user defined values
                keys["features_values"][p] = v["Values"]

    def promote_features_to_design(self, feature_list):
        """
        Promote a feature to a design parameter, adding dimensions to the exploration space
        """
        keys = self["parameters"]

        if "optimization_features" not in keys:
            keys["optimization_features"] = []

        # Parse every entry in the dictionary
        # And append their value/type to the feature map
        for p_name in feature_list:
            if p_name not in self["parameters"]["features_type"]:
                raise Exception(
                    'Tried to promote unknown parameter "{}" to an optimization feature, '
                    "parameters should first be promoted to feature beforehand"
                )
            keys["optimization_features"].append(p_name)

    @deprecated(
        "This getter is deprecated and will be removed in future versions, use obj.objectives instead"
    )
    def get_objectives(self):
        """
        Return the list of objectives to optimize
        """
        if "experiment" not in self or "objectives" not in self["experiment"]:
            raise Exception("No objectives set")
        return self["experiment"]["objectives"]

    @deprecated(
        "This getter is deprecated and will be removed in future versions, use obj.design_parameters instead"
    )
    def get_design_features(self):
        """
        Return a list of design features (Features that will be optimized)
        """
        return self["parameters"]["optimization_features"]

    @deprecated(
        "This getter is deprecated and will be removed in future versions, use obj.parameters_type instead"
    )
    def get_all_features_types(self):
        """
        Return a dict containing the type of every feature defined in the experiment,
        whether they are design features or input
        """
        return self["parameters"]["features_type"]

    @deprecated(
        "This getter is deprecated and will be removed in future versions, use obj.input_parameters instead"
    )
    def get_kernel_input_features(self):
        """
        Returns a list of the features that will be used as input to the kernel
        """
        return self["parameters"]["inputs"]

    @deprecated(
        "This getter is deprecated and will be removed in future versions, use obj.compilation_configuration instead"
    )
    def get_compilation_configuration(self):
        """
        Returns the compilation configuration of the experiment
        May be None if the experiment has no compilation features
        """
        return self.compilation_configuration

    # Redefine common operators so the configuration can be used like a dictionary
    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    # Overload the [] operator to access/set keys
    def __getitem__(self, item):
        return self.keys[item]

    def __setitem__(self, key, value):
        self.keys[key] = value

    def __contains__(self, item):
        return item in self.keys
