"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Optional
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scipy.stats import norm
import pprint
import logging

"""
This file contains a scalable implementation of a variance estimator using Quantile Regression with LightGBM.

The variance is estimated using the interquantile range in the following way:
    variance = ((pred_hb - pred_lb) / (2 * z)) ** 2

Where:
- where pred_hb and pred_lb are the upper and lower bounds of the quantile predictions
- z is the quantile value for the given alpha (e.g. 0.05 for 95% confidence interval)

Two methods are implemented:
- Standard: one model for the lower quantile and one for the upper quantile
- Forced symmetry: a single model is used to compute the upper quantile,
and the lower quantile is computed symmetrically around the mean.
This method enforces normality required for the variance estimation and is cheaper to compute.
"""

logger = logging.getLogger(__name__)


def _validate_lightgbm_params(params: dict, alternative_params: dict = None) -> Optional[dict]:
    """Validate LightGBM parameters.

    :param params: LightGBM parameters to validate.
    :type params: dict
    :param alternative_params: Alternative parameters to use if validation fails, defaults to None
    :type alternative_params: dict, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: Optional[dict]
    """
    if params is None:
        return alternative_params.copy()

    try:
        LGBMRegressor(**params)
    except ValueError as e:
        msg = "The parameters provided are not valid for the LGBMRegressor:"
        msg += pprint.pformat(params)
        raise ValueError(msg) from e

    return dict(params)


class IQRVarianceEstimator:
    """
    Mixtures of LGBMRegressor for scalable mean and variance regression.
    """

    # Default parameters for the quantile regressors
    default_quantile_lgbm_params = {
        "n_estimators": 800,
        "n_jobs": -1,
        "objective": "quantile",
        "min_data_in_leaf": 80,
        "boosting": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 80,
        "verbose": -1,
    }

    # Default parameters for the mean/median regressors
    # We use the same parameters for now
    default_lgbm_params = default_quantile_lgbm_params.copy()

    def __init__(
        self,
        quantile_lgb_params: Optional[dict] = None,
        mean_lgb_params: Optional[dict] = None,
        alpha: float = 0.95,
        method: str = "standard",
    ):
        """Initialize the IQRVarianceEstimator.

        :param quantile_lgb_params: LightGBM parameters for the quantile regressors, defaults to None
        :type quantile_lgb_params: Optional[dict], optional
        :param mean_lgb_params: LightGBM parameters for the mean/median regressors, defaults to None
        :type mean_lgb_params: Optional[dict], optional
        :param alpha: The quantile level to fit the models, defaults to 0.95
        :type alpha: float, optional
        :param method: The method to use for variance estimation, defaults to "standard"
            - If "standard", two independent quantile models are used for the lower and upper quantiles (alpha and 1-alpha)
            and variance is estimated through the interquantile range.
            - If "forced_symmetry", a single quantile model is used for the provided quantile (alpha),
            and we use a median model to compute variance
        :type method: str, optional
        :raises ValueError: If the parameters are invalid
        :raises ValueError: If the objective is not "quantile" for quantile regression
        :raises ValueError: If the method is not recognized
        """

        self._quantile_params = _validate_lightgbm_params(quantile_lgb_params, self.default_quantile_lgbm_params)

        if self._quantile_params["objective"] != "quantile":
            raise ValueError("The objective must be 'quantile' for quantile regression")

        self._lgb_params = _validate_lightgbm_params(mean_lgb_params, self.default_lgbm_params)

        if method not in ["standard", "forced_symmetry"]:
            raise ValueError("The 'method' parameter must be either 'standard' or 'forced_symmetry'")

        self.method = method
        self._alpha_lgbm = None
        self._inverse_alpha_lgbm = None
        self._mean_lgbm = None
        self._median_lgbm = None

        if not (0 < alpha < 1):
            raise ValueError("The 'alpha' parameters must be between 0 and 1")

        # In standard mode, where we build two independent models for the lower and upper quantiles,
        # We always work with the upper quantile (alpha > 0.5)
        if method == "standard" and alpha < 0.5:
            logger.warning(
                f"LGBMMixture was provided with a lower quantile alpha {alpha}. "
                f"Standard mode requires the upper quantile."
                f"Converting to upper quantile form {1 - alpha}"
            )
            alpha = 1 - alpha

        # If we are in forced symmetry mode, we do not require alpha > 0.5 as we can use the lower/upper
        # quantile symmetrically around the median
        self.alpha = alpha

        self.mean_only = False

        # Internal variables to store the ordering and encoding of the features
        self._ordering = None
        self._encoding = None

    # We wrap both the quantile and mean parameters to ensure they are always valid
    @property
    def quantile_params(self) -> dict:
        return self._quantile_params

    @quantile_params.setter
    def quantile_params(self, value: dict) -> None:
        self._quantile_params = _validate_lightgbm_params(value, self.default_quantile_lgbm_params)
        if self._quantile_params["objective"] != "quantile":
            raise ValueError("The objective must be 'quantile' for quantile regression")

    @property
    def lgb_params(self) -> dict:
        return self._lgb_params

    @lgb_params.setter
    def lgb_params(self, value: dict) -> None:
        self._lgb_params = _validate_lightgbm_params(value, self.default_lgbm_params)

    # We define all models as properties to prevent writes from outside the class
    @property
    def alpha_lgbm(self) -> LGBMRegressor:
        return self._alpha_lgbm

    @property
    def inv_alpha_lgbm(self) -> Optional[LGBMRegressor]:
        return self._inverse_alpha_lgbm

    @property
    def mean_lgbm(self) -> LGBMRegressor:
        return self._mean_lgbm

    @property
    def median_lbm(self) -> LGBMRegressor:
        return self._median_lgbm

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # We need to ensure that the columns are always in the same order
        # Because LightGBM does not check the column names
        self._ordering = sorted(X.columns)
        X = X[self._ordering]

        self._encoding = X.dtypes.to_dict()

        def _fit_model(X: pd.DataFrame, y: pd.Series, params: dict, override_params: dict) -> LGBMRegressor:
            # Copy the parameters to avoid modifying the original dictionary
            params = dict(params)

            # Override any parameters that need to be changed for this specific model
            params.update(override_params)
            model = LGBMRegressor(**params)
            model.fit(X, y)
            return model

        self._alpha_lgbm = _fit_model(X, y, self.quantile_params, {"alpha": self.alpha})
        self._mean_lgbm = _fit_model(X, y, self.lgb_params, {"objective": "mse"})

        # We always need the median model:
        # - In standard mode, to enforce symmetry in case of crossing quantiles
        # - In forced symmetry mode, to compute the symmetric quantile around the median
        self._median_lgbm = _fit_model(X, y, self.lgb_params, {"objective": "mae"})

        # When in standard mode, we also need the upper quantile model
        if self.method == "standard":
            self._inverse_alpha_lgbm = _fit_model(X, y, self.quantile_params, {"alpha": 1 - self.alpha})
        else:
            self._inverse_alpha_lgbm = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensured the input DataFrame as the correct dtypes

        :param df: The DataFrame to change the types on
        :type df: pd.DataFrame
        :return: A correctly typed DataFrame
        :rtype: pd.DataFrame
        """

        # LightGBM will complain if the DataFrame doesn't have the right dtypes
        res = df.astype(self._encoding)
        return res[self._ordering]

    def _standard(self, X: pd.DataFrame) -> np.ndarray:
        if self._inverse_alpha_lgbm is None or self._alpha_lgbm is None:
            raise ValueError(
                "Missing models for standard quantile variance estimation: " "lower or upper quantile model is None"
            )

        pred_hb = self._alpha_lgbm.predict(X)
        pred_lb = self._inverse_alpha_lgbm.predict(X)

        crossing_index = pred_hb < pred_lb
        if np.any(crossing_index):
            logger.warning(
                f"Quantile crossing detected for {np.sum(crossing_index)} samples out of {len(X)}."
                "Enforcing symmetry around the median."
            )
            # In case of crossing quantiles, we enforce symmetry
            # around the median prediction
            X_cross = X[crossing_index]
            median = self._median_lgbm.predict(X_cross)

            pred_hb[crossing_index] = median + (median - pred_lb[crossing_index])

        sigma = (pred_hb - pred_lb) / (2 * norm.ppf(self.alpha))

        return sigma**2

    def _forced_symmetry(self, X: pd.DataFrame) -> np.ndarray:
        if self._alpha_lgbm is None or self._median_lgbm is None:
            raise ValueError(
                "Missing models for forced symmetry quantile variance estimation: " "upper quantile or median model is None"
            )

        pred_hb = self._alpha_lgbm.predict(X)
        median = self._median_lgbm.predict(X)

        sigma = (pred_hb - median) / norm.ppf(self.alpha)

        return sigma**2

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        X = self._encode(X)

        mean = self._mean_lgbm.predict(X)

        if self.mean_only:
            return mean

        if self.method == "standard":
            variance = self._standard(X)
        elif self.method == "forced_symmetry":
            variance = self._forced_symmetry(X)
        else:
            raise ValueError(f"Unknown method {self.method}")

        # Ensure the variance is non-negative
        variance = np.maximum(variance, 0)
        return mean, variance
