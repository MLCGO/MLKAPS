"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
from typing import Callable, Generator, Iterable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from mlkaps.modeling.model_wrapper import ModelWrapper


class OptunaModelTuner:
    """
    Base class for all model tuners based on Optuna
    """

    known_tuners = {}

    def __init__(
        self,
        inputs: pd.DataFrame,
        labels: pd.Series | Iterable,
        metric: Callable[[Iterable, Iterable], float] = mean_absolute_error,
        n_folds: int = 10,
    ):
        """_summary_

        :param inputs: The data to fit the model on
        :type inputs: pd.DataFrame
        :param labels: The label to fit the model for
        :type labels: pd.Series | Iterable
        :param metric: A function (sklearn metric) that computes an error/score for the model,
        defaults to mean_absolute_error. This function will be minimized
        :type metric: Callable[[Iterable, Iterable], float], optional
        :param n_folds: The number of folds to use to compute the model score, defaults to 5
        :type n_folds: int, optional
        """
        self.inputs = inputs
        self.labels = labels
        self.metric = metric
        self.n_folds = n_folds
        self.study = None

    def __init_subclass__(cls, model_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if model_name is None:
            model_name = cls.__name__
        cls.known_tuners[model_name] = cls

    def _get_hyperparameters(self, trial: optuna.trial) -> dict:
        """Generat the model hyperparameters for the current optuna trial

        :param trial: The current optuna trial
        :type trial: optuna.trial
        :return: A dictionnary containing the hyperparameters for the model
        :rtype: dict
        """
        raise NotImplementedError()

    def _build_model(self, hyperparameters: dict) -> ModelWrapper:
        """Build and fit a model using the given hyperparameters

        :param parameters: The hyperparameters to build the model with
        :type parameters: dict
        :return: A fitted model
        :rtype: ModelWrapper
        """
        raise NotImplementedError()

    def _kfold_evaluate(self, model: ModelWrapper) -> Generator[tuple[list[float], float], None, None]:
        """Fit and evalaue the model using kfold. Yields the midway score after every fold.

        :param model: A fitted model
        :type model: ModelWrapper
        :yield: A generator that outptuts the midway score at every fold
        :rtype: Generator[tuple[list[float], float], None, None]
        """

        scores = []

        # Build the folds
        kf = KFold(min(self.n_folds, len(self.inputs)), shuffle=True)

        # Append to the scores and compute the midway score
        for train_idx, test_idx in kf.split(self.inputs):
            model.fit(self.inputs.iloc[train_idx], self.labels.iloc[train_idx])
            predictions = model.predict(self.inputs.iloc[test_idx])

            # In some weird cases, especially with low number of samples
            # LightGBM can return NaN predictions
            if np.isnan(predictions).any():
                continue

            scores.append(self.metric(self.labels.iloc[test_idx], predictions))
            midway_score = np.mean(scores)

            yield scores, midway_score

    def _objective(self, trial: optuna.trial) -> float:
        """Entry point for the optuna study

        :param trial: The current optuna trial
        :type trial: optuna.trial
        :raises optuna.TrialPruned: Raised if the trial must be pruned
        :return: The score of the model, lower is better
        :rtype: float
        """
        parameters = self._get_hyperparameters(trial)
        model = self._build_model(parameters)

        scores = None
        for scores, midway_score in self._kfold_evaluate(model):
            trial.report(midway_score, len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        # If the model only returned NaN predictions, return infinity
        if scores is None or len(scores) == 0:
            return np.inf

        # If any of the scores is NaN, return infinity
        res = np.mean(scores)
        if np.isnan(res):
            return np.inf
        return res

    def run(self, time_budget: int = 60, n_trials: int = None) -> tuple[ModelWrapper, dict]:
        """
        Run the optuna tuner until the time budget is expired or n_trials have been run, whichever comes first.

        :param time_budget: The time allowed for tuning, in seconds, defaults to 60
        :type time_budget: int, optional
        :param n_trials: The maximum number of trials for optuna, defaults to None
        :type n_trials: int, optional
        :return: A fitted and tuned model, and the best parameters found
        :rtype: tuple[ModelWrapper, dict]
        """

        # Build the description for the progress bar
        time_budget_desc = f"{time_budget}s" if time_budget is not None else None
        trials_desc = f"{n_trials} trials" if n_trials is not None else None
        budget_desc = ", ".join(filter(None, [time_budget_desc, trials_desc]))

        with tqdm(desc=f"Running optuna for {budget_desc}", unit=" trial", leave=None) as pbar:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", pruner=optuna.pruners.SuccessiveHalvingPruner())
            self.study = study
            study.optimize(
                self._objective,
                timeout=time_budget,
                n_trials=n_trials,
                callbacks=[lambda *_: pbar.update(1)],
            )

        params = self._get_hyperparameters(study.best_trial)
        return self._build_model(params), params


class OptunaRecorder(OptunaModelTuner):
    """
    Decorator that will dump the optuna study to a file
    """

    def __init__(self, wrapee: OptunaModelTuner, record_path: str | pathlib.Path | None = None):
        """
        Initialize the recorder.

        :param wrapee: The tuner to decorate.
        :type wrapee: OptunaModelTuner
        :param record_path: The path to record to as a .csv file, defaults to None.
        :type record_path: str | pathlib.Path | None, optional
        """
        if isinstance(record_path, str):
            record_path = pathlib.Path(record_path)

        self.record_path = record_path
        self.tuner = wrapee

    def _get_session_id(self) -> int:
        """
        Fetch the current tuning session id from the output path, 0 if the file doesn't exists yet

        :return: _description_
        :rtype: int
        """
        if not self.record_path.exists():
            return 0

        data = pd.read_csv(self.record_path)
        return data["training_id"].max() + 1

    def _build_record(self, study: optuna.Study, session_id: int) -> pd.DataFrame:
        """
        Build a DataFrame that records all the optuna trials for this tuning session

        :param study: The optuna study to record
        :type study: optuna.Study
        :param session_id: The id of this tuning session
        :type session_id: int
        :return: A DataFrame containing all records
        :rtype: pd.DataFrame
        """
        res = pd.DataFrame(
            [[session_id, i, v.value, len(self.tuner.inputs)] for i, v in enumerate(study.get_trials())],
            columns=["training_id", "iteration", "score", "number_of_samples"],
        )

        res = pd.concat([res, pd.DataFrame([v.params for v in study.get_trials()])], axis=1)
        return res

    def _record(self, study: optuna.study):
        """
        Record the study to a file in .csv format.

        :param study: The study to record.
        :type study: optuna.study
        """
        self.record_path.parent.mkdir(parents=True, exist_ok=True)

        session_id = self._get_session_id()
        records = self._build_record(study, session_id)

        if self.record_path.exists():
            records = pd.concat([pd.read_csv(self.record_path), records])

        records.to_csv(self.record_path, index=False)

    def run(self, *args, **kwargs) -> tuple[ModelWrapper, dict]:
        """
        Execute the tuner and record the study

        :return: A fitted and tuned model, and the best parameters found
        :rtype: tuple[ModelWrapper, dict]
        """

        try:
            model, params = self.tuner.run(*args, **kwargs)
        finally:
            if self.tuner.study is not None:
                self._record(self.tuner.study)

        return model, params
