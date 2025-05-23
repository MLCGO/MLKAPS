"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os
from pathlib import Path

import numpy as np
from deprecated import deprecated

from ._parsing_helpers import set_if_defined

# This file provides utility functions to parse the configuration file
# And translate the mains parameters into a form that is easier to use
# The configuration is split into sections,
# with sections corresponding to different tasks inside mlkaps
# The configuration file is a json file, with the following structure:
# {
#   "SECTION1": {
#       "key1": "value1",
#       "key2": "value2",
#       ...
#   },
#   "SECTION2": { ...
# }


class ParserError(Exception):
    pass


# Parse the visualization section of the json section if any
# Otherwise, create a blank visualization section with default parameters
@deprecated
def _maybe_parse_visualization(configuration, mapped_section, json_section, output_path="", prefix="", dpi=100):
    mapped_section["visualization"] = {}
    visu = mapped_section["visualization"]

    exp = configuration.output_directory()
    # If the output path is not absolute, create a new directory
    visu["output_path"] = exp + output_path if not os.path.isabs(output_path) else output_path
    visu["prefix"] = prefix
    visu["dpi"] = dpi

    # Default every scale to linear
    visu["scales"] = {p: "lin" for p in configuration.get_kernel_input_features()}
    # FIXME: Not all visualization needs the step size, recheck this
    visu["steps"] = {p: 100 for p in configuration.get_kernel_input_features()}

    if "visualization" not in json_section:
        Path(visu["output_path"]).mkdir(parents=True, exist_ok=True)
        return

    json_section = json_section["visualization"]

    special_features = [i for i in configuration.get_kernel_input_features() if i in json_section]
    for param in special_features:
        feature_parameters = json_section[param]
        if "scale" in feature_parameters:
            visu["scales"][param] = feature_parameters["scale"]
        if "step" in feature_parameters:
            visu["steps"][param] = feature_parameters["step"]

    set_if_defined("output_path", json_section, visu)
    # Check if the user provided an absolute path
    if not os.path.isabs(visu["output_path"]):
        visu["output_path"] = exp + visu["output_path"]

    set_if_defined("prefix", json_section, visu)
    set_if_defined("dpi", json_section, visu)

    Path(visu["output_path"]).mkdir(parents=True, exist_ok=True)


# Parses the design parameters from the json file
def _maybe_parse_design_parameters(configuration, json_section):
    # If the design parameters are undefined, just skip this step
    if "DESIGN_PARAMETERS" not in json_section:
        raise Exception("The configuration file does not define any design parameters")

    keys = configuration["parameters"]
    design_parameter = json_section["DESIGN_PARAMETERS"]
    keys["design"] = list(design_parameter)
    configuration.add_features(design_parameter)
    configuration.promote_features_to_design(design_parameter)


# Parses the inputs of the experiment
def _parse_kernel_inputs(configuration, json_section):
    if "KERNEL_INPUTS" not in json_section:
        raise Exception("The configuration file does not define any input parameters")

    keys = configuration["parameters"]

    kernel_inputs = json_section["KERNEL_INPUTS"]
    keys["inputs"] = list(kernel_inputs.keys())
    configuration.add_features(kernel_inputs)


# Parses the PARAMETERS section of the json file
# Including design, kernel inputs and compilation flags parameters
def parse_experiment_parameters(configuration, json_file):
    # Create the "parameters" section in the dict
    configuration["parameters"] = {}
    keys = configuration["parameters"]
    keys["features_type"] = {}
    keys["features_values"] = {}

    experiment_parameters = json_file["PARAMETERS"]

    # Check for either design or compilation flags parameters
    _maybe_parse_design_parameters(configuration, experiment_parameters)
    _parse_kernel_inputs(configuration, experiment_parameters)


# Parse the objectives and data sampling parameters
def parse_objectives(configuration, data_section):
    experiment_section = data_section["EXPERIMENT"]
    if "objectives" not in experiment_section:
        raise Exception("The configuration file does not define any objectives")

    configuration["experiment"] = {}
    keys = configuration["experiment"]

    keys["objectives"] = experiment_section["objectives"]

    keys["objectives_list"] = []
    keys["objectives_directions"] = {}
    keys["objectives_bounds"] = {}

    # Check if keys["objectives"] is a list (support legacy objectives declaration without bounds and directions)
    if isinstance(keys["objectives"], list):
        for objective in keys["objectives"]:
            # Assuming details are provided in another section or default values
            keys["objectives_list"].append(objective)
            keys["objectives_directions"][objective] = "minimize"  # Default direction
            keys["objectives_bounds"][objective] = np.nan  # Default bound
    elif isinstance(keys["objectives"], dict):
        for objective, details in keys["objectives"].items():
            keys["objectives_list"].append(objective)
            keys["objectives_directions"][objective] = details.get("direction", "minimize")
            keys["objectives_bounds"][objective] = details.get("bound", np.nan)
    else:
        raise Exception("Invalid format for objectives")


def parse_modeling(configuration, json_data):
    # Create a blank modeling section in the map
    configuration["modeling"] = {}

    cmap = configuration["modeling"]

    data_modelling = json_data["MODELING"]

    cmap["modeling_method"] = data_modelling["modeling_method"]
    cmap["parameters"] = data_modelling.get("parameters", {})


def parse_clustering(configuration, json_data):
    # clustering related parameters
    clustering_section = json_data["CLUSTERING"]

    configuration["clustering"] = {}
    cmap = configuration["clustering"]

    cmap["clustering_method"] = clustering_section["clustering_method"]
    cmap["clustering_parameters"] = {}

    set_if_defined("clustering_parameters", clustering_section, cmap, {})
