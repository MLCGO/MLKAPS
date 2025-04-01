"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.modeling.model_wrapper import ModelWrapper
from mlkaps.modeling.optuna_model_tuner import OptunaModelTuner
import pandas as pd
import lightgbm
import optuna


class LightGBMWrapper(ModelWrapper, wrapper_name="lightgbm"):
    """
    Wrapper for LightGBM regressor.

    Ensures that the features are passed in the correct order and are correctly typed
    """

    def __init__(self, **hyperparameters):
        """
        Initialize a new LightGBM model. The model is built lazily.
        """

        super().__init__(**hyperparameters)
        self.model = None

        # We build the model lazily
        self.encoding = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensured the input DataFrame as the correct dtypes

        :param df: The DataFrame to change the types on
        :type df: pd.DataFrame
        :return: A correctly typed DataFrame
        :rtype: pd.DataFrame
        """

        # LightGBM will complain if the DataFrame doesn't have the right dtypes
        res = df.astype(self.encoding)
        return res

    def _fit(self, X, y):
        # Save the dtypes of the training dataset
        self.encoding = {k: v for k, v in zip(X.columns, X.dtypes)}

        # Lazily build the model
        if self.model is None:
            self.model = lightgbm.LGBMRegressor(**self.hyperparameters)
        self.model.fit(X[self.ordering], y)

    def predict(self, inputs: pd.DataFrame):
        inputs = self._encode(inputs)
        return self.model.predict(inputs[self.ordering])

    def set_max_thread(self, n_threads: int):
        """Restrict the maximum number of threads allowed for the LightGBM model

        :param n_threads: The maximum allowed number of threads
        :type n_threads: int
        """

        self.model.set_params({"n_jobs": n_threads})


class OptunaTunerLightgbm(OptunaModelTuner, model_name="lightgbm"):
    """
    Specialization of OptunaModelTuner to tune LightGBM models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Generate the model parameters using the optuna trial

        :param trial: The current optuna trial
        :type trial: optuna.Trial
        :return: A dictionnary containing the hyperparameters of the model
        :rtype: dict
        """

        res = {
            "objective": "mae",
            "verbose": -1,
            "n_jobs": -1,
            "boosting": trial.suggest_categorical("boosting", ["gbdt", "rf"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1),
            "n_estimators": trial.suggest_int("n_estimators", 1, 600),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, min(512, len(self.inputs) - 1)),
        }

        if len(self.inputs) > 10:
            res["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.1, 1.0)

        return res

    def _build_model(self, hyperparameters: dict) -> LightGBMWrapper:
        """Build a LightGBM model using the given parameters

        :param parameters: The hyperparameters of the model
        :type parameters: dict
        :return: The fitted model
        :rtype: LightGBMWrapper
        """

        res = LightGBMWrapper(**hyperparameters)
        res.fit(self.inputs, self.labels)
        return res
