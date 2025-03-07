from mlkaps.modeling.model_wrapper import ModelWrapper
import pandas as pd
import xgboost


class XGBoostModelWrapper(ModelWrapper, wrapper_name="xgboost"):
    """
    Wrapper for XGBoost regressor.

    Ensures that the features are passed in the correct order and are correctly typed
    """

    def __init__(self, **hyperparameters):
        """
        Initialize a new XGBoost model. The model is built lazily.
        """
        super().__init__(**hyperparameters)
        self.model = None
        self.encoding = None
        self.categorical_features = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the input DataFrame has the correct dtypes

        :param df: The DataFrame to change the types on
        :type df: pd.DataFrame
        :return: A correctly typed DataFrame
        :rtype: pd.DataFrame
        """
        res = df.astype(self.encoding)
        return res

    def _fit(self, X, y):
        # Save the dtypes of the training dataset
        self.encoding = {k: v for k, v in zip(X.columns, X.dtypes)}

        # Identify categorical features
        self.categorical_features = [i for i, dtype in enumerate(X.dtypes) if dtype.name == "category"]

        # Lazily build the model
        if self.model is None:
            self.hyperparameters["enable_categorical"] = True
            self.model = xgboost.XGBRegressor(**self.hyperparameters)
        self.model.fit(X[self.ordering], y, eval_metric="rmse")

    def predict(self, inputs: pd.DataFrame):
        inputs = self._encode(inputs)
        return self.model.predict(inputs[self.ordering])

    def set_max_thread(self, n_threads: int):
        """Restrict the maximum number of threads allowed for the XGBoost model

        :param n_threads: The maximum allowed number of threads
        :type n_threads: int
        """
        self.model.set_params(n_jobs=n_threads)
