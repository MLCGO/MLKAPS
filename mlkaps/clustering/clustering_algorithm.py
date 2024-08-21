"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
from mlkaps.configuration import ExperimentConfig
import logging


def _make_model(
    configuration: ExperimentConfig, feature: str, optimization_results: pd.DataFrame
) -> DecisionTreeClassifier | DecisionTreeRegressor:
    """
    Factory method for building the clustering model depending on the feature type and
    experiment configuration

    :param configuration: The configuration of the experiment to build the surrogate for
    :type configuration: ExperimentConfig
    :param feature: The name of the feature to build the surrogate for
    :type feature: str
    :param optimization_results: A dataframe containing the optimum configurations
    :type optimization_results: pd.DataFrame
    :raises ValueError: Raised if we didn't find the appropriate model type to handle the given feature
    :return: The model to use for clustering
    :rtype: DecisionTreeClassifier | DecisionTreeRegressor
    """

    feature_type = configuration.parameters_type[feature]

    model_type = None
    if feature_type in ["Categorical", "Boolean"]:
        model_type = DecisionTreeClassifier
    elif feature_type in ["int", "float"]:
        model_type = DecisionTreeRegressor
    else:
        raise ValueError(
            f"Failed to find clustering model for '{feature}' ('{feature_type}')"
        )

    clustering_parameters = configuration["clustering"]["clustering_parameters"]
    model = model_type(**clustering_parameters)

    # We need to ensure boolean are correctly encoded as bool, and not string, objects, or integers as this raises an exception in sklearn
    if feature_type == "Boolean":
        optimization_results[feature] = optimization_results[feature].astype(bool)

    # Now fit the model on the whole dataset
    model.fit(
        optimization_results[configuration.input_parameters],
        optimization_results[feature],
    )

    return model


def _generate_dummy_model(
    configuration: ExperimentConfig,
    optimization_results: pd.DataFrame,
    optim_feature: str,
) -> DummyClassifier:
    """
    Helper function to generate a dummy model when only one configuration value is used for all inputs

    :param configuration: The configuration of the experiment to build the surrogates for
    :type configuration: ExperimentConfig
    :param optimization_results: A dataframe containing the results of the optimization phase
    :type optimization_results: pd.DataFrame
    :param optim_feature: The feature to build the model for
    :type optim_feature: str
    :return: A dummy model from sklearn
    :rtype: DummyClassifier
    """

    logging.warning(
        f"Clustering model for feature '{optim_feature}' is constant, "
        f"using a dummy model"
    )
    model = DummyClassifier(strategy="most_frequent")
    model.fit(
        optimization_results[configuration.input_parameters],
        optimization_results[optim_feature],
    )
    return model


def generate_clustering_models(
    configuration: ExperimentConfig, optimization_results: pd.DataFrame
) -> dict[str, DecisionTreeClassifier | DecisionTreeRegressor]:
    """
    Train models to predict the best design parameters for each feature using the optimization
    results.

    :param configuration: The configuration of the experiment to build the surrogates for
    :type configuration: ExperimentConfig
    :param optimization_results: A dataframe containing the results of the optimization phase
    :type optimization_results: pd.DataFrame
    :return: A dictionnary of decision tree per feature.
        The type of the model can change depending on the type of the feature
    :rtype: dict[str, DecisionTreeClassifier | DecisionTreeRegressor]
    """

    models = {}

    # Create/train a model for every design parameter
    for optim_feature in tqdm(
        configuration.design_parameters, desc="Building decision trees", leave=None
    ):

        # If the feature is constant, we output a dummy model that always returns the same value
        # This is necessary since decision trees cannot handle constant features
        if len(np.unique(optimization_results[optim_feature])) == 1:
            model = _generate_dummy_model(
                configuration, optimization_results, optim_feature
            )
        else:
            model = _make_model(configuration, optim_feature, optimization_results)

        output_path = configuration.output_directory / (
            optim_feature + "_clustered_model.pkl"
        )
        with open(output_path, "wb") as f:
            pickle.dump(model, f)

        models[optim_feature] = model

    return models
