"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import textwrap
import logging
from typing import Iterable
import pprint
import pandas as pd
from tqdm import tqdm

from .common import KernelSamplingError


class _ProgressBarWrapper:

    def __init__(self, progress_bar: object | bool, nsamples=None):
        # If the progress bar is an object, we assume it either has an update method or is callable
        # And that we shouldn't close it (i.e. the user may pass a progress bar with 1000 items but we will only use 10)
        if not isinstance(progress_bar, bool):
            self._progress_bar = progress_bar
            self._closure = None
            self._update = getattr(progress_bar, "update", None) or progress_bar
            return

        # Else if the progress bar is a boolean, we create a new progress bar
        if progress_bar:
            self._progress_bar = tqdm(total=nsamples, desc="Sampling", leave=None)
            self._closure = self._progress_bar.close
            self._update = self._progress_bar.update
        else:
            self._progress_bar = None
            self._closure = None
            self._update = None

    def __call__(self):
        if self._update is not None:
            self._update()

    def close(self):
        if self._closure is not None:
            self._closure()

    def __repr__(self):
        return f"ProgressBarWrapper(progress_bar={self.progress_bar})"


class MonoKernelExecutor:

    def __init__(
        self,
        runner,
        resolver,
        samples_checkpoint,
        *,
        progress_bar: bool | object = False,
        pre_execution_callbacks: None | Iterable[callable] = None,
    ):
        self.runner = runner
        self.resolver = resolver
        self.samples_checkpoint = samples_checkpoint
        self.progress_bar = progress_bar

        self.pre_execution_callbacks = pre_execution_callbacks
        self._ran_pre_execution = False

    def _maybe_do_pre_execution(self):
        if self._ran_pre_execution or self.pre_execution_callbacks is None:
            return

        logging.debug("Running pre-execution callbacks")

        for callback in self.pre_execution_callbacks:
            callback()

        self._ran_pre_execution = True

    def _log_error(self, result):
        if result.timed_out:
            msg_prefix = "Timeout occurred during sample execution:\n"
        else:
            msg_prefix = "Error occurred during sample execution:\n"

        # Print the parameters and the error
        msg_suffix = textwrap.indent(pprint.pformat(result.data), "\t| ")
        msg_suffix += "\n" + textwrap.indent(pprint.pformat(result.error), "\t> ")

        logging.error(msg_prefix + msg_suffix)

    def _run_all_samples(self, samples: pd.DataFrame) -> pd.DataFrame:
        n_failures = 0
        results = None

        pbar = _ProgressBarWrapper(self.progress_bar, len(samples))

        batch_size = min(10, len(samples))
        batch = []
        # itertuples returns each row as a tuple with the types of each column unchanged
        for count, row in enumerate(samples.itertuples(index=False), start=1):
            sample = row._asdict()  # note this is a private interface to a Pandas object.
            result = self.runner(sample)

            if result.error is not None:
                n_failures += 1
                self._log_error(result)

                if n_failures > 100:
                    raise KernelSamplingError("Too many sampling failures, aborting")

            batch.append(sample | result.data)
            pbar()

            if count % batch_size == 0:
                results = self._decorate_resolve_and_save_batch(samples, results, batch)
                batch = []

        if count % batch_size != 0:
            results = self._decorate_resolve_and_save_batch(samples, results, batch)

        pbar.close()
        return results

    def _decorate_resolve_and_save_batch(
        self, samples: pd.DataFrame, results: pd.DataFrame, batch: list[dict]
    ) -> pd.DataFrame:

        # Convert batch to a dataframe, append it to the .csv file, append it to the current
        # results, and return.
        batch = pd.DataFrame(batch)

        # Normalize the types -- I don't think this is needed, but no harm
        dtypes = {col: dtype for col, dtype in batch.dtypes.items()}
        batch = batch.astype(dtypes)

        # note the column order is determined when we save the batch in save_batch.
        batch = self.resolver(batch).reset_index(drop=True)
        batch = self.samples_checkpoint.save_batch(batch)

        if results is None:
            return batch
        else:
            return pd.concat([results, batch], ignore_index=True)

    def __call__(self, samples: None | pd.DataFrame) -> pd.DataFrame:

        if samples is None or len(samples) == 0:
            return None

        self._maybe_do_pre_execution()

        results = self._run_all_samples(samples)
        return results
