"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

"""
Helper functions to build Sampler objects
"""

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
