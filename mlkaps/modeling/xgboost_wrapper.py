"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.modeling.model_wrapper import ModelWrapper
from mlkaps.configuration import ExperimentConfig
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost


class XGBoostModelWrapper(ModelWrapper, wrapper_name="xgboost"):
    """
    Wrapper for XGBoostRegressor from xgboost. Note that this model uses sklearn API,
    but requires categorical columns to be one-hot encoded
    """

    def __init__(self, config: ExperimentConfig, objective, inputs=None):
        super().__init__(config, objective, inputs=inputs)
        model_params = config["modeling"]["xgboost_regressor_parameters"]
        self.model = xgboost.XGBRegressor(**model_params)
        self.one_hot_encoder = None

    def _build_onehot_encoder(self, X):
        X = X[self.ordering]

        # Extract the categorical columns
        categorical_columns = [
            feature
            for feature in self.ordering
            if self.configuration.parameters_type[feature] == "Categorical"
        ]

        encoder = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
            remainder="passthrough",
        )
        self.encoder = encoder.fit(X)

    def _encode(self, X) -> pd.DataFrame:
        return self.encoder.transform(X[self.ordering])

    def _fit(self, X, y):
        self._build_onehot_encoder(X)
        X = self._encode(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._encode(X)
        return self.model.predict(X)

    def set_max_thread(self, n_threads: int):
        self.model.set_params({"n_jobs": n_threads})
