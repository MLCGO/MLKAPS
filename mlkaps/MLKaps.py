#!/usr/bin/env python3
"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import json
import pathlib
import sys
from argparse import ArgumentParser
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import logging
import time
from mlkaps.clustering import generate_clustering_models
from mlkaps.codegen.codegen import decision_trees_to_c
from mlkaps.configuration import ExperimentConfig
from mlkaps.modeling.modeling import build_main_surrogates
from mlkaps.optimization.genetic_optimizer import (
    GeneticOptimizer,
    GeneticOptimizerConfig,
)
from mlkaps.sample_collection.kernel_sampling import sample_kernel
from ._utils import setup_logging, add_file_logger, ensure_not_root
import os


def run_clustering(args, experiment_config, optim_results):
    """
    Build the clustering models (decision trees) based on optimal configurations.

    The clustering models will be serialized to the output directory specified in the experiment_config
    parameter using pickle.

    :param args: The CLI arguments, as returned by ArgumentParser
    :param experiment_config: The ExperimentConfig object containing the
        parsed experiment configuration
    :type experiment_config: mlkaps.configuration.ExperimentConfig
    :param optim_results: A pandas DataFrame containing the optimum configurations
        to build the clustering models from
    :type optim_results: pandas.DataFrame
    """

    # TODO: quick-restart was removed for the clustering module as it relied on pickle
    # Properly re-implement it
    clustered_models = generate_clustering_models(experiment_config, optim_results)

    # plot_all_decisions_maps(experiment_config, optim_results, clustered_models)
    # plot_all_decision_tree(experiment_config, clustered_models)

    # Experimental
    decision_trees_to_c(experiment_config, clustered_models)


def run_optimization(args, config_dict, experiment_config, surrogate_models):
    """
    Run an optimizer using the provided surrogate_models, and return a Pandas.DataFrame
    containing the optimum configurations found.

    The optimum are commonly computed on a grid, specified in the configuration file.

    :param args: The CLI arguments, as returned by argparse.ArgumentParser
    :param config_dict: A dict containing the configuration parsed from json
        This is used to access parameters that may not be stored in the experiment_config object
    :type config_dict: dict
    :param experiment_config: The ExperimentConfig object containing the
        parsed experiment configuration
    :type experiment_config: mlkaps.configuration.ExperimentConfig
    :param surrogate_models: A dict containing one surrogate model per objective
    :type surrogate_models: dict

    :return: The optimum configurations found by the optimizer
    :rtype: pandas.DataFrame
    """

    qr = args.quick_restart
    has_quick_restart = qr in ["clustering"]

    genetic_config = GeneticOptimizerConfig.from_configuration_dict(config_dict, experiment_config)

    optim_results = {}
    if has_quick_restart:
        logging.info("Quick-restart: Loading the optimization results from previous run")
        path = pathlib.Path(experiment_config.output_directory / "optim.csv")
        if not path.exists():
            raise FileNotFoundError(path)
        optim_results = pd.read_csv(path)
    else:
        gen_optim = GeneticOptimizer(genetic_config, surrogate_models)
        optim_results = gen_optim.run()

    return optim_results


def create_surrogate_models(args, kernel_sampling_output, experiment_config):
    """
    Train one surrogate model per objective using the samples provided

    :param args: The CLI arguments, as returned by argparse.ArgumentParser
    :param kernel_sampling_output: A DataFrame containing samples to train the surrogates on.
        The training data must contain the same features described in experiment_config.
    :type kernel_sampling_output: pandas.DataFrame
    :param experiment_config: The ExperimentConfig object containing the
        parsed experiment configuration
    :type experiment_config: mlkaps.configuration.ExperimentConfig

    :return: One surrogate model per objective
    :rtype: dict(str, mlkaps.modeling.modeling.ModelWrapper)
    """

    surrogate_models = build_main_surrogates(experiment_config, kernel_sampling_output)

    return surrogate_models


def run_kernel_sampling(args, experiment_config, config_dict):
    """
    Runs the sampling module according to the provided experiment configuration

    :param args: The CLI arguments, as returned by argparse.ArgumentParser
    :param experiment_config: The ExperimentConfig object containing the
        parsed experiment configuration
    :type experiment_config: mlkaps.configuration.ExperimentConfig
    :param config_dict: A dict containing the configuration parsed from json
        This is used to access parameters that may not be stored in the experiment_config object
    :type config_dict: dict

    :return: The sampled points and their measured objectives values
    :rtype: pandas.DataFrame
    """

    qr = args.quick_restart

    has_quick_restart = qr in ["modeling", "optimization", "clustering"]

    if has_quick_restart:
        logging.info("Quick-restart: Loading the samples from previous run")
        path = (experiment_config.output_directory / "kernel_sampling/samples.csv").resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
    else:
        df = sample_kernel(experiment_config, config_dict)

    return df


def parse_configuration_file(config_path, args):
    """
    Parse the configuration file stored in config_path

    :param config_path: The path to the experiment configuration file (json file)
    :param args: The program arguments

    :return: The raw dictionnary containing the configuration, and the parsed configuration
    :rtype: (dict, mlkaps.configuration.ExperimentConfig)
    """

    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)

    experiment_config = ExperimentConfig.from_dict(config_dict, Path(config_path).parent, args.output_directory)

    return config_dict, experiment_config


def run_pipeline(args, config_dict, experiment_config):
    """
    Run the MLKAPS pipeline based on the experiment configuration

    :param args: The program arguments
    :param config_dict: The raw dictionnary containing the experiments
    :param experiment_config: The ExperimentConfig object containing the
        parsed experiment configuration
    :type experiment_config: mlkaps.configuration.ExperimentConfig
    """

    ensure_not_root(args)

    with logging_redirect_tqdm():
        logging.info("Auto-tuning pipeline started")
        begin = time.time()

        kernel_sampling_results = run_kernel_sampling(args, experiment_config, config_dict)

        surrogate_models = create_surrogate_models(args, kernel_sampling_results, experiment_config)

        optim_results = run_optimization(args, config_dict, experiment_config, surrogate_models)

        run_clustering(args, experiment_config, optim_results)
        end = time.time()
        logging.info(f"Pipeline ended, took {end - begin} seconds")


def parse_arguments(argv, configuration_required=False):
    """
    Parse the program arguments

    :param argv: An array containing the CLI arguments.
        Can also be manually built for testing purposes

    :return: The values of the parsed arguments
    :rtype: argparse.Namespace
    """

    parser = ArgumentParser(
        prog="MLKAPS",
        description="Machine Learning for Kernel Accuracy and Performance Studies -- Auto-tuning tool for HPC kernels",
    )
    if configuration_required:
        parser.add_argument(
            "configuration_file",
            help="Path to the configuration file of the experiment",
        )

    parser.add_argument(
        "--quick-restart",
        choices=["sampling", "modeling", "optimization", "clustering"],
        help="Restart MLKAPS on the given step",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        help="Path to the output directory. Overrides any output path defined in the configuration file.",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set logging to debug level",
    )

    parser.add_argument(
        "--allow-root",
        action="store_true",
        help="Allow running as root. By default, MLKAPS will exit and report an error to avoid security issues.",
    )

    args = parser.parse_args(args=argv)

    if args.quick_restart is not None:
        logging.info(f"Trying to quick-restart from '{args.quick_restart}'")

    # We usually don't allow running MLKAPS in sudo mode
    # However, it may be required when running MLKAPS inside a docker
    # And when inside github workflows
    # This check for an environment variable to bypass the check
    if os.environ.get("MLKAPS_IS_IN_DOCKER") == "True":
        args.allow_root = True

    return args


def maybe_enable_debug(args):
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)


def setup_output_dir(config_dict, experiment_config, args):
    metadata_dir = experiment_config.metadata_directory
    add_file_logger(metadata_dir)

    config_path = metadata_dir / "configuration.json"

    logging.debug(f"Copying configuration to {config_path}")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)


def run_from_config(argv, config_dict, working_dir, output_directory=None):
    """
    Run mlkaps from a given configuration dictionary, skipping the json loading

    :param argv: A list of arguments to pass to the program, as if called from the command line
    :type argv: list
    :param config_dict: A dictionnary containing the raw experiment configuration
    :type config_dict: dict
    :param working_dir: The working directory to use for the experiment
    """

    setup_logging()
    args = parse_arguments(argv)
    maybe_enable_debug(args)

    experiment_config = ExperimentConfig.from_dict(config_dict, working_dir, output_directory=output_directory)

    setup_output_dir(config_dict, experiment_config, args)
    run_pipeline(args, config_dict, experiment_config)


def run_from_args(argv=None):
    """
    Run mlkaps using command line-like arguments

    :param argv: A list of arguments to pass to the program, as if called from the command line
    """
    setup_logging()

    if argv:
        sys.argv = argv

    args = parse_arguments(argv, configuration_required=True)

    maybe_enable_debug(args)

    config_path = args.configuration_file
    config_dict, experiment_config = parse_configuration_file(config_path, args)

    setup_output_dir(config_dict, experiment_config, args)
    run_pipeline(args, config_dict, experiment_config)


if __name__ == "__main__":
    run_from_args()
