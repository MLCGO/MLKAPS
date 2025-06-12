"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os

from mlkaps.configuration.experiment_configuration import ExperimentConfig
from mlkaps.sample_collection.samples_checkpoint import SamplesCheckpoint
from mlkaps.sample_collection.kernel_sampling import sample_kernel

import pytest
import pandas as pd


class Test2D:
    # test checkpointing of 'samples.csv'

    def _run_helper(self, tmp_path, builder_helper, config_name):

        # get basic information
        curr_path = os.path.abspath(os.path.dirname(__file__))
        builder_helper.setup(srcpath=curr_path, build=False)

        working_dir = builder_helper.srcpath
        output_dir = tmp_path

        # set up to run sampling
        config_dict, _ = builder_helper.load_configuration(config_name, tmp_path)
        experiment_config = ExperimentConfig.from_dict(config_dict, working_dir, output_directory=output_dir)
        samples_checkpoint = SamplesCheckpoint(
            experiment_config.output_directory, experiment_config.parameters_type, experiment_config.objectives
        )

        # delete the checkpoint and create again to make sure we have clean start
        samples_checkpoint.delete_file()
        samples_checkpoint = SamplesCheckpoint(
            experiment_config.output_directory, experiment_config.parameters_type, experiment_config.objectives
        )

        # Collect samples, make sure file and in-memory are equal
        s0 = sample_kernel(experiment_config, config_dict, samples_checkpoint)
        samples_checkpoint.consistency_check(s0)

        # # Sample again.  The file exists, so no samples should actually be run.
        s1 = sample_kernel(experiment_config, config_dict, samples_checkpoint)
        samples_checkpoint.consistency_check(s1)
        samples_checkpoint.consistency_check(s0)

        # # Delete the file, and sample again.  Due to randomness, the new samples
        # # should be different from those collected above.
        samples_checkpoint.delete_file()
        samples_checkpoint = SamplesCheckpoint(
            experiment_config.output_directory, experiment_config.parameters_type, experiment_config.objectives
        )

        s2 = sample_kernel(experiment_config, config_dict, samples_checkpoint)
        samples_checkpoint.consistency_check(s2)
        # assert s0, s2 are not equal
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(s0, s2, rtol=1e-5)

    @pytest.mark.parametrize(
        "file",
        [
            "hvs_short.json",
            "multilevel_hvsr_short.json",
            "random_short.json",
            "ga_adaptive_short.json",
        ],
    )
    def test_config(self, tmp_path, builder_helper, file):
        self._run_helper(tmp_path, builder_helper, file)
