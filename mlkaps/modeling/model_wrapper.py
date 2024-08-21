"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

from mlkaps.configuration import ExperimentConfig
import pandas as pd
from typing import Iterable


# This file provides functions to encode data and train a model on it.
class ModelWrapper:
    """
    Base class for model wrappers. Model wrappers are used to encapsulate the logic required to
    train/use a specific model, which avoid propagating this logic elsewhere in the code.

    Note that most libraries offer a scikit-learn compatible API, so in future versions we might
    be able to reduce the required amount of wrappers by offering a generic wrapper.
    """

    known_models = {}

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.ordering = None

    def __init_subclass__(cls, wrapper_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if wrapper_name is None:
            wrapper_name = cls.__name__
        cls.known_models[wrapper_name] = cls

    def _fit(self, X: pd.DataFrame, y: Iterable):
        """Implementation detail for the fit function

        :param X: The data to fit the model on
        :type X: pd.DataFrame
        :param y: The label to fit the model for
        :type y: Iterable
        """
        raise NotImplementedError()

    def fit(self, X: pd.DataFrame, y: Iterable):
        """Track the ordering of the parameters, and call _fit(...)

        :param X: The data to fit the model on
        :type X: pd.DataFrame
        :param y: The label to fit the model for
        :type y: Iterable
        """

        self.ordering = sorted(list(X.columns))
        self._fit(X, y)

    def predict(self, X: pd.DataFrame) -> Iterable:
        """Use the model to predict the label

        :param X: The data to predict for
        :type X: pd.DataFrame
        :return: The predictions of the model
        :rtype: Iterable
        """
        raise NotImplementedError()

    def set_max_thread(n_threads: int):
        """Limit the maximum number of threads of the model, if possible

        :param n_threads: The number of threads to use
        :type n_threads: int
        """
        raise NotImplementedError()


class ModelThreadLimiter:
    """
    Context manager to limit the number of threads available to one or multiple models.
    Models are restored to use all threads afterward
    """

    def __init__(self, models: Iterable[ModelWrapper] | ModelWrapper, max_threads=1):
        """Initialize the context

        :param models: One or multiple models to limit the threads on
        :type models: Iterable[ModelWrapper] | ModelWrapper
        :param max_threads: The target number of threads, defaults to 1
        :type max_threads: int, optional
        """

        if not isinstance(models, Iterable):
            models = [models]
        self.models = models
        self.max_threads = max_threads

    def __enter__(self):
        """Limit all models to the defined number of threads"""
        for m in self.models:
            m.set_max_thread(self.max_threads)

    def __exit__(self, type, value, traceback):
        """
        Exit the context and allow the models to use all available threads

        :param type: unused
        :param value: unused
        :param traceback: unused
        """

        for m in self.models:
            m.set_max_thread(-1)

        return False
