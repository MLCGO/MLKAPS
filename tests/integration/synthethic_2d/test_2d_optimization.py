"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os

from mlkaps.MLKaps import parse_arguments, ExperimentConfig, run_kernel_sampling, create_surrogate_models
from mlkaps.optimization.optimizer_checkpoint import OptimizerCheckpoint
from mlkaps.optimization.genetic_optimizer import GeneticOptimizer, GeneticOptimizerConfig
import pytest


class Test2D:
    # test checkpointing of 'optim.csv'

    def _run_helper(self, tmp_path, builder_helper, config_name):

        # get basic information
        curr_path = os.path.abspath(os.path.dirname(__file__))
        builder_helper.setup(srcpath=curr_path, build=False)

        working_dir = builder_helper.srcpath
        output_dir = tmp_path

        # set up for the ml-kaps pipeline
        config_dict, _ = builder_helper.load_configuration(config_name, output_dir)
        args = parse_arguments([])
        experiment_config = ExperimentConfig.from_dict(config_dict, working_dir, output_directory=output_dir)

        # run up to optimization step
        kernel_sampling_results = run_kernel_sampling(args, experiment_config, config_dict)
        surrogate_models = create_surrogate_models(args, kernel_sampling_results, experiment_config)

        # create optimizer checkpoint
        optimizer_checkpoint = OptimizerCheckpoint(experiment_config.output_directory)

        # delete the checkpoint file to make sure we have clean start
        optimizer_checkpoint.delete_file()

        # run optimization
        genetic_config = GeneticOptimizerConfig.from_configuration_dict(config_dict, experiment_config)
        gen_optim = GeneticOptimizer(genetic_config, surrogate_models, optimizer_checkpoint)
        optim_results1 = gen_optim.run()
        optimizer_checkpoint.consistency_check(optim_results1)

        # run again without deleting the file
        optim_results2 = gen_optim.run()
        optimizer_checkpoint.consistency_check(optim_results2)
        optimizer_checkpoint.consistency_check(optim_results1)

        # delete some rows from the saved_file and run again.
        # note opening the file with 'w' clears it.
        new_len = int(len(optim_results2) / 2)
        prefix_results3 = optim_results2[0:new_len]
        with open(optimizer_checkpoint.output_path, "w"):
            pass
        prefix_results3.to_csv(optimizer_checkpoint.output_path, index=False)

        # the samples are unchanged, so we should reproduce the same results
        # for the deleted rows
        optim_results3 = gen_optim.run()
        optimizer_checkpoint.consistency_check(optim_results3)
        optimizer_checkpoint.consistency_check(optim_results2)
        optimizer_checkpoint.consistency_check(optim_results1)

    @pytest.mark.parametrize(
        "file",
        [
            "random.json",
        ],
    )
    def test_config(self, tmp_path, builder_helper, file):
        self._run_helper(tmp_path, builder_helper, file)
