"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

"""
This file contains a scalable implementation of a variance estimator using Quantile Regression with LightGBM.

The variance is estimated using the interquantile range in the following way:
    variance = ((pred_hb - pred_lb) / (2 * z)) ** 2

Where:
- where pred_hb and pred_lb are the upper and lower bounds of the quantile predictions
- z is the quantile value for the given alpha (e.g. 0.05 for 95% confidence interval)

Two methods are implemented:
- Standard: one model for the lower quantile and one for the upper quantile
- Forced symmetry: a single model is used to compute the upper quantile, and the lower quantiel is computed symmetrically around the mean.
This method enforces normality required for the variance estimation and is cheapr to compute.

"""

from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scipy.stats import norm
import pprint
import logging

logger = logging.getLogger(__name__)

class QuantileLGBMMixture:
    """
    Mixtures of LGBMRegressor for scalable mean and variance regression.
    """

    def __init__(self, params: dict = None, hb_alpha: float = 0.05, method="stabard", mean_only=False):

        if params is not None:
            try:
                LGBMRegressor(**params)
            except ValueError as e:
                msg = "The parameters provided are not valid for the LGBMRegressor. Please check the parameters:"
                msg += pprint.pformat(params)
                raise ValueError(msg) from e
        else:
            params = {
                "n_estimators": 800,
                "n_jobs": -1,
                "objective": "quantile",
                "min_data_in_leaf": 80,
                "boosting": "gbdt",
                "learning_rate": 0.01,
                "num_leaves": 80,
                "verbose": -1,
            }
        
        assert params["objective"] == "quantile", "The objective must be quantile for the LGBMRegressor"
        self.params = params

        assert 0 < hb_alpha < 1, "The alpha must be between 0 and 1"
        assert method in ["forced_symmetry", "standard"], "The method must be either forced_symmetry or standard"

        self.lbm = None
        self.hbm = None
        self.pred = None
        self.lb_alpha = 1 - hb_alpha
        self.hb_alpha = hb_alpha
        self.ordering = None
        self.mean_only = False

    def train(self, X: pd.DataFrame, y, refit=False):
        ordering = sorted(X.columns)
        X = X[ordering]
        self.ordering = ordering

        if self.method == "standard":
            params_lb = self.parameters.copy()
            params_lb["alpha"] = self.lb_alpha

            self.lbm = LGBMRegressor(**params_lb)
            self.lbm.fit(X, y)

        params_hb = self.parameters.copy()
        params_hb["alpha"] = self.hb_alpha

        self.hbm = LGBMRegressor(**params_hb)
        self.hbm.fit(X, y)

        params_pred = self.parameters.copy()
        params_pred["objective"] = "mae"
        self.pred = LGBMRegressor(**params_pred)
        self.pred.fit(X, y)

    def _standard(self, X: pd.DataFrame, pred):
        pred_hb = self.hbm.predict(X)
        delta = abs(pred_hb - pred)
        # Just ensure the ordering is correct
        pred_lb = pred - delta
        pred_hb = pred + delta

        return pred_lb, pred_hb
    
    def _old_variance(self, X: pd.DataFrame, pred):
        pred_hb = self.hbm.predict(X)
        pred_lb = self.lbm.predict(X)
        # Just ensure the ordering is correct
        pred_lb = np.minimum(pred_lb, pred)
        pred_hb = np.maximum(pred_hb, pred)

        return pred_lb, pred_hb
    
    def predict(self, X: pd.DataFrame):
        X = X[self.ordering]

        pred = self.pred.predict(X)
        if self.mean_only:
            return pred

        if self.method == "standard":
            pred_lb, pred_hb = self._standard(X, pred)
        else:
            pred_lb, pred_hb = self._old_variance(X, pred)

        # Sanity check to ensure the predictions are within the bounds
        # of the quantiles
        violations = ~((pred_lb <= pred) & (pred <= pred_hb))
        if np.any(violations):
            logger.warning(
                f"Some predictions are outside the bounds of the quantiles: {violations.sum()} / {len(violations)}"
                f"Some variance and mean predictions will be invalid."
            )

        z = norm.ppf(self.hb_alpha)
        sigma = ((pred_hb - pred_lb) / (2 * z)) ** 2
        variance = np.maximum(sigma, 0)
        return pred, variance
