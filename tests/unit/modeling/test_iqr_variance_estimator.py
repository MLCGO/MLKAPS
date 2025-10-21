from mlkaps.modeling.iqr_variance_estimator import IQRVarianceEstimator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape


def _generate_dgp(count, nfolds=2):

    x1 = [0, 20]

    points = np.random.uniform(x1[0], x1[1], count)

    def fmeans(x):
        fold = int(x // (x1[1] / nfolds))
        return x1[0] + (x1[1] - x1[0]) * (fold + 0.5) / nfolds

    def fsigma(x):
        # We consider a standard deviation of 5% of the mean
        return 0.05 * x

    cmeans = np.vectorize(fmeans)(points)
    csigmas = np.vectorize(fsigma)(cmeans)

    y = np.random.normal(cmeans, csigmas)

    res = pd.DataFrame({"x1": points, "y": y, "true_means": cmeans, "true_sigma": csigmas})
    return res


class TestIqrVarianceEstimator:

    def test_can_build_simple(self):
        quantile_params = {
            "objective": "quantile",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        mean_params = {
            "objective": "mse",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "verbose": -1,
        }
        model = IQRVarianceEstimator(quantile_lgb_params=quantile_params, mean_lgb_params=mean_params, alpha=0.975)
        assert model is not None

        df = _generate_dgp(2500, nfolds=2)

        model.fit(df[["x1"]], df["y"])
        mu_hat, var_hat = model.predict(df[["x1"]])

        mape_mu = mape(df["true_means"], mu_hat)

        # We should have good accuracy on both mean and variance
        # In this simple case
        assert mape_mu < 0.15

        rel_std_error = mape(df["true_sigma"], np.sqrt(var_hat))
        # If the relative error if below 15%, we are good enough
        assert rel_std_error < 0.15

        corr_std = np.corrcoef(np.sqrt(var_hat), df["true_sigma"])[0, 1]
        assert corr_std > 0.8

    def test_can_build_forced_symmetry(self):
        quantile_params = {
            "objective": "quantile",
            "n_estimators": 120,
            "learning_rate": 0.08,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        mean_params = {
            "objective": "mse",
            "n_estimators": 120,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "verbose": -1,
        }
        model = IQRVarianceEstimator(
            quantile_lgb_params=quantile_params, mean_lgb_params=mean_params, alpha=0.975, method="forced_symmetry"
        )
        assert model is not None

        df = _generate_dgp(2500, nfolds=2)

        model.fit(df[["x1"]], df["y"])
        mu_hat, var_hat = model.predict(df[["x1"]])

        mape_mu = mape(df["true_means"], mu_hat)

        # We should have good accuracy on both mean and variance
        # In this simple case
        assert mape_mu < 0.15

        rel_std_error = mape(df["true_sigma"], np.sqrt(var_hat))
        # If the relative error if below 20%, we are good enough
        # Note that we use a higher threshold than in standard mode
        # as forced symmetry is less robust
        assert rel_std_error < 0.20

        corr_std = np.corrcoef(np.sqrt(var_hat), df["true_sigma"])[0, 1]
        assert corr_std > 0.8

    def test_honors_variance_contract(self):
        quantile_params = {
            "objective": "quantile",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        model = IQRVarianceEstimator(quantile_lgb_params=quantile_params, mean_lgb_params=quantile_params, alpha=0.975)
        assert model is not None

        df = _generate_dgp(1200, nfolds=2)
        model.fit(df[["x1"]], df["y"])

        # In standard mode, we should have model.alpha > 0.5 (== 0.975 here)
        assert model.alpha > 0.5
        assert model.alpha_lgbm is not None
        model_params = model.alpha_lgbm.get_params()
        assert model_params["objective"] == "quantile"
        assert model_params["alpha"] == 0.975

        # And the inverse quantile model should also be defined
        assert model.inv_alpha_lgbm is not None
        model_params = model.inv_alpha_lgbm.get_params()
        assert model_params["objective"] == "quantile"
        assert model_params["alpha"] == 1 - 0.975

        model = IQRVarianceEstimator(
            quantile_lgb_params=quantile_params, mean_lgb_params=quantile_params, alpha=0.25, method="forced_symmetry"
        )
        model.fit(df[["x1"]], df["y"])

        # In forced symmetry, the model should not alter the provided alpha
        assert model.alpha == 0.25
        assert model.alpha_lgbm is not None
        model_params = model.alpha_lgbm.get_params()
        assert model_params["objective"] == "quantile"
        assert model_params["alpha"] == 0.25

        # The inverse quantile model should not be defined in forced symmetry mode
        assert model.inv_alpha_lgbm is None

    def test_can_set_parameters(self):
        quantile_params = {
            "objective": "quantile",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        mean_params = {
            "objective": "mse",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "verbose": -1,
        }
        model = IQRVarianceEstimator(
            quantile_lgb_params=quantile_params, mean_lgb_params=mean_params, alpha=0.975, method="forced_symmetry"
        )
        assert model is not None
        assert model.method == "forced_symmetry"
        assert model.alpha == 0.975

        df = _generate_dgp(1200, nfolds=2)
        model.fit(df[["x1"]], df["y"])

        assert model.alpha_lgbm is not None
        # LightGBM will set some default parameters, so we check only the ones we set
        model_params = model.alpha_lgbm.get_params()
        assert all(item == model_params[key] for key, item in quantile_params.items())

        assert model.mean_lgbm is not None
        model_params = model.mean_lgbm.get_params()
        assert all(item == model_params[key] for key, item in mean_params.items())

        assert model.median_lbm is not None
        model_params = model.median_lbm.get_params()
        mean_params["objective"] = "mae"
        assert all(item == model_params[key] for key, item in mean_params.items())

        # Should be none in forced symmetry mode
        assert model.inv_alpha_lgbm is None

    def test_mean_only(self):
        quantile_params = {
            "objective": "quantile",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "min_data_in_leaf": 100,
            "verbose": -1,
        }
        model = IQRVarianceEstimator(quantile_lgb_params=quantile_params, mean_lgb_params=quantile_params, alpha=0.975)
        assert model is not None

        df = _generate_dgp(1200, nfolds=2)

        model.fit(df[["x1"]], df["y"])

        model.mean_only = True
        mu_hat = model.predict(df[["x1"]])

        # Assert that the predict method returns only the mean
        assert isinstance(mu_hat, np.ndarray)
        assert mu_hat.shape == (df.shape[0],)
