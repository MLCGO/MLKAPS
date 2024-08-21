"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import pickle
import pandas as pd
import logging
import textwrap
import pprint
from typing import Iterable

from mlkaps.configuration import ExperimentConfig
from mlkaps.modeling.model_wrapper import ModelWrapper
from mlkaps.modeling.optuna_model_tuner import OptunaModelTuner, OptunaRecorder
from mlkaps.modeling.encoding import encode_dataframe


class ModelingError(Exception):
    pass


class SurrogateFactory:
    """Factory class to easily builds surrogate models according to the configuration"""

    def __init__(
        self, config: ExperimentConfig, sampled_data: pd.DataFrame, model_name=None
    ):
        self.config = config

        # Ensure the data is correctly encoded to the right type
        self.sampled_data = encode_dataframe(config.parameters_type, sampled_data)

        # Fetch the modeling method in the configuration if not given
        if model_name is None:
            model_name = self.config["modeling"]["modeling_method"]
        self.model_name = model_name

    def _build_optuna_tuner(self, config: dict, X: pd.DataFrame, y: Iterable):
        """Build an optuna tuner according to the passed configuration

        :param config: The configuration for the optuna model tuner
        :type config: dict
        :param X: The input of the model
        :type X: pd.DataFrame
        :param y: The target/objective of the model
        :type y: Iterable
        :raises ModelingError: Raised if no tuner could be find for the  model type specified in the configuration
        """

        # First try fetch the correct tuner
        model_name = config["model_name"]
        tuner = OptunaModelTuner.known_tuners.get(model_name, None)
        if tuner is None:
            raise ModelingError("Could not find optuna tuner with name '{model_name}'")

        # Get the tuning budget
        time_budget = config.get("time_budget")
        n_trials = config.get("n_trials")

        if time_budget is None and n_trials is None:
            logging.warning(
                "No budget was set for optuna, defaulting to 10 minutes per tuning session"
            )
            time_budget = 10 * 60

        tuner = tuner(X, y)

        # Check if we should record the tuning session
        if config.get("record", True):
            tuner = OptunaRecorder(
                tuner, self.config.output_directory / f"optuna_records_for_{model_name}"
            )

        return tuner, time_budget, n_trials

    def _build_model_using_optuna(
        self, config: dict, X: pd.DataFrame, y: Iterable
    ) -> ModelWrapper:
        """Build a tuned model using optuna

        :param config: The configuration for the optuna model tuner
        :type config: dict
        :param X: The input of the model
        :type X: pd.DataFrame
        :param y: The target/objective of the model
        :type y: Iterable
        :return: A tuned and fitted model
        :rtype: ModelWrapper
        """

        config = config["parameters"]

        tuner, time_budget, n_trials = self._build_optuna_tuner(config, X, y)

        model, params = tuner.run(time_budget=time_budget, n_trials=n_trials)

        msg = textwrap.indent(pprint.pformat(params), "\t")
        logging.info(f"Finished building model with optuna, parameters are\n{msg}")

        return model

    def _build_model_using_parameters(
        self, model_name: str, config: dict, X: pd.DataFrame, y: Iterable
    ) -> ModelWrapper:
        """Build a model using the default hyperparameters or one present in the configuration

        :param model_name: The name of the model to use
        :type model_name: str
        :param config: The configuration of the model
        :type config: dict
        :param X: The input of the model
        :type X: pd.DataFrame
        :param y: The target/objective of the model
        :type y: Iterable
        :raises ModelingError: Raised if no model was found with the given name
        :return: The fitted model
        :rtype: ModelWrapper
        """

        model = ModelWrapper.known_models.get(model_name, None)

        if model is None:
            raise ModelingError(
                f"Could not find model wrapper with name '{model_name}'"
            )

        config = config["parameters"]
        model = model(**config)
        model.fit(X, y)
        return model

    def build(
        self, objective: str, inputs: Iterable[str] = None, model_name: str = None
    ) -> ModelWrapper:
        """Build a new model for the given objective

        :param objective: The objective/label to fit the model on
        :type objective: str
        :param inputs: A list of features to fit the model on, defaults to None
        :type inputs: Iterable[str], optional
        :param model_name: The name of the model type to use, defaults to None
        :type model_name: str, optional
        :return: A fitted model
        :rtype: ModelWrapper
        """

        if model_name is None:
            model_name = self.model_name

        if inputs is None:
            inputs = list(self.config.parameters_type.keys())

        config = self.config["modeling"]

        X = self.sampled_data[inputs]
        y = self.sampled_data[objective]

        if model_name == "optuna":
            surrogate = self._build_model_using_optuna(config, X, y)
        else:
            surrogate = self._build_model_using_parameters(model_name, config, X, y)

        return surrogate


def build_main_surrogates(experiment_config, kernel_sampling_output) -> dict:
    """
    Factory function for building one surrogate per objective in the experiment configuration.
    """
    surrogate_models = {}
    modeling_method = experiment_config["modeling"]["modeling_method"]
    factory = SurrogateFactory(
        experiment_config, kernel_sampling_output, modeling_method
    )

    for obj in experiment_config.objectives:
        surrogate_models[obj] = factory.build(obj)
        with open(
            experiment_config.output_directory / (str(obj) + "_model.pkl"), "wb"
        ) as f:
            pickle.dump(surrogate_models[obj], f)

    return surrogate_models
