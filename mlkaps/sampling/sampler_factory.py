"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

"""
Helper functions to build Sampler objects
"""

from deprecated import deprecated
from mlkaps.configuration.experiment_configuration import ExperimentConfig


class SamplerFactory:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def from_config(self, sampler_name, config_dict: dict = {}):
        # Temporary hack to avoid circular inclusions
        from .generic_bounded_sampler import RandomSampler, LhsSampler
        from .grid_sampler import GridSampler
        from .adaptive import HVSampler, MultilevelHVS
        from .sampler import SamplerError

        mapping = {
            "lhs": LhsSampler,
            "random": RandomSampler,
            "grid": GridSampler,
            "hvs": HVSampler,
            "multilevel_hvs": MultilevelHVS,
        }

        if sampler_name not in mapping:
            raise SamplerError(f"Unknown sampler: {sampler_name}")

        return mapping[sampler_name](**config_dict)


@deprecated
def build_sampler_from_dict(config: dict):
    """
    Factory method to build a sampler from a standardized configuration dictionary.
    Note that the sampled variables are not set by this method, and must be set using the
    set_variables(...) method before sampling.


    :param config_dict:
        A dictionary containing the configuration of the sampler. The dictionary must contain the
        following keys:
        - "sampling_method": The name of the sampling method to use. Currently supported methods
        are "grid", "random", "lhs" and "adaptive".

        if sampling_method == "adaptive":
        - The "parameters" key must be defined, containing:
            * "type": The type of adaptive sampling method to use. Currently supported methods are
            "hvs" and "multilevel_hvs".

            * (Optional) "method_parameters": A dictionary containing the parameters of the sampling
            method. The parameters depend on the sampling method used. The dict is directly passed
            to the constructor of the sampling method.

    :type config_dict: dict

    :return:
        The sampler defined by the configuration dictionary.

    :raise SamplerError: An exception is raised if the configuration dictionary is invalid, or the sampling method is
        unknown.
    """

    sampling_method = config["sampling_method"]

    # Check if the sampler has parameters defined in the config dict
    # If not, set the parameters to an empty dict, else, the parameters will be forwarded
    # to the sampler constructor

    sampler_parameters = config.get("method_parameters", {})

    match sampling_method:
        case "grid":
            return GridSampler(**sampler_parameters)
        case "random":
            return RandomSampler(**sampler_parameters)
        case "lhs":
            return LhsSampler(**sampler_parameters)
        case "adaptive":
            # FIXME: Improve the dict layout to unify the names of each section
            # We should only use two keys: "sampling_method" and "method_parameters"
            # sampling method should be "HVS" rather than "adaptive" + "type": "hvs"
            sampler_parameters = {}
            if "method_parameters" in config["parameters"]:
                sampler_parameters = config["parameters"]["method_parameters"]

            match config["parameters"]["type"]:
                case "hvs":
                    return HVSampler(**sampler_parameters)
                case "multilevel_hvs":
                    return MultilevelHVS(**sampler_parameters)
                case _:
                    raise SamplerError(
                        f"Unknown adaptive sampling method: "
                        f"{config['parameters']['type']}"
                    )
        case _:
            raise SamplerError(f"Unknown sampling method: {sampling_method}")
