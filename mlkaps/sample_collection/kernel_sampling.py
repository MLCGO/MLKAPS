"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
import pandas as pd
import logging
from tqdm import tqdm
import pprint
import textwrap

from mlkaps.configuration import ExperimentConfig
from mlkaps.sampling.sampler_factory import SamplerFactory
from mlkaps.sampling.adaptive.ga_adaptive import GAAdaptiveSampler
from mlkaps.sampling.adaptive import (
    StoppingCriterionFactory,
    AdaptiveSamplingOrchestrator,
)

from .mono_kernel_executor import MonoKernelExecutor
from .function_harness import MonoFunctionHarness, FunctionPath
from .subprocess_harness import MonoSubprocessHarness
from .failed_run_resolver import DiscardResolver, ConstantResolver


def _get_key_or_error(cdict: dict, key: str, error_msg: str | None = None, default=None, fatal=True):
    if key not in cdict:
        error_msg = error_msg or f"Missing parameter '{key}'"
        res = default
        if fatal:
            msg = textwrap.indent(f"Dictionnary:\n{pprint.pformat(cdict)}\n", "\t=> ")
            msg = f"{error_msg}\n{msg}"
            raise ValueError(msg)
        else:
            logging.warning(error_msg)
    else:
        res = cdict[key]
    return res


def _make_path_absolute(path: pathlib.Path | str, reference: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)

    if path.is_absolute():
        return path

    return reference / path


class _StaticSamplerInterfaceWrapper:
    """
    Sampler that uses a static spatial-sampler to generate a list of samples that will be executed
    in a one-shot manner
    """

    def __init__(
        self,
        kernel_sampler,
        config: ExperimentConfig,
        sampler_type: str,
        config_dict: dict,
        output_path: pathlib.Path,
    ):
        self.kernel_sampler = kernel_sampler

        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.config = config
        self.sampler_type = sampler_type
        self.config_dict = config_dict
        self.output_path = output_path

    def __call__(self) -> pd.DataFrame:
        sampler, n_samples = self._build_sampler()
        return self._sample(sampler, n_samples)

    def _build_sampler(self):

        sampler = SamplerFactory(self.config).from_config(self.sampler_type, self.config_dict.get("sampler_parameters", {}))

        if hasattr(sampler, "set_variables"):
            variables_types = self.config.parameters_type
            variables_values = self.config.feature_values
            sampler.set_variables(variables_types, variables_values)

        n_samples = _get_key_or_error(self.config_dict, "nsamples")
        return sampler, n_samples

    def _sample(self, sampler, n_samples):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        samples = sampler.sample(n_samples)

        kBatchSize = 100
        batches = min(kBatchSize, len(samples))
        res = []
        # Run the samples in batches in order to periodically save the results to file
        with tqdm(total=len(samples), desc="Running samples", leave=None) as pbar:
            self.kernel_sampler.progress_bar = pbar
            for i in range(0, len(samples), batches):
                batch = samples.iloc[i : i + batches]
                results = self.kernel_sampler(batch)
                res.append(results)

                tmp = pd.concat(res, axis=0).reset_index(drop=True)
                tmp.to_csv(self.output_path, index=False)

        res = tmp
        return res


class _GAAdaptiveInterfaceWrapper:

    def __init__(
        self,
        kernel_sampler,
        config: ExperimentConfig,
        sampler_type: str,
        config_dict: dict,
        output_path: pathlib.Path,
    ):
        self.kernel_sampler = kernel_sampler

        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.config = config
        self.config_dict = config_dict
        self.output_path = output_path

    def __call__(self) -> pd.DataFrame:
        sampler = self._build_sampler()
        res = sampler()
        res.to_csv(self.output_path, index=False)
        return sampler()

    def _build_sampler(self):
        n_samples = _get_key_or_error(self.config_dict, "n_samples")
        bootstrap_ratio = _get_key_or_error(self.config_dict, "bootstrap_ratio")
        initial_ga_ratio = _get_key_or_error(self.config_dict, "initial_ga_ratio")
        final_ga_ratio = _get_key_or_error(self.config_dict, "final_ga_ratio")

        samples_per_iteration = self._get_samples_per_iter(n_samples, bootstrap_ratio)

        do_early_stopping = _get_key_or_error(
            self.config_dict,
            "do_early_stopping",
            "Enabling early stopping for the GA by default",
            default=True,
            fatal=False,
        )
        use_optuna = _get_key_or_error(
            self.config_dict,
            "use_optuna",
            "Disabling optuna by default",
            default=False,
            fatal=False,
        )

        sampler = GAAdaptiveSampler(
            self.kernel_sampler,
            self.config,
            n_samples,
            samples_per_iteration,
            bootstrap_ratio,
            initial_ga_ratio,
            final_ga_ratio,
            do_early_stopping=do_early_stopping,
            use_optuna=use_optuna,
        )
        return sampler

    def _get_samples_per_iter(self, n_samples, bootstrap_ratio):
        n_iterations = self.config_dict.get("n_iterations")
        samples_per_iteration = self.config_dict.get("samples_per_iteration")

        # Compute the number of samples taken with GA at each iteration
        # Done either via a direct fixed number of samples or a fixed number of iterations
        if all([n_iterations is None, samples_per_iteration is None]):
            msg = textwrap.indent(f"Dictionnary:\n{pprint.pformat(self.config_dict)}\n", "\t=> ")
            msg = f"Options 'n_iterations' and 'samples_per_iteration' are mutually exclusive\n{msg}"
            raise ValueError(msg)
        elif n_iterations is not None:
            # if a number of iteration is given, compute the number of samples per iteration
            # by dividing the number of samples by the number of iterations
            samples_per_iteration = n_samples * (1 - bootstrap_ratio) / n_iterations
        elif samples_per_iteration is None:
            msg = textwrap.indent(f"Dictionnary:\n{pprint.pformat(self.config_dict)}\n", "\t=> ")
            msg = f"Neither 'n_iterations' or 'samples_per_iteration' were defined\n{msg}"
            raise ValueError(msg)
        return samples_per_iteration


class _AdaptiveSamplerInterfaceWrapper:
    def __init__(
        self,
        kernel_sampler,
        config: ExperimentConfig,
        sampler_type: str,
        config_dict: dict,
        output_path: pathlib.Path,
    ):
        self.kernel_sampler = kernel_sampler

        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.config = config
        self.sampler_type = sampler_type
        self.config_dict = config_dict
        self.output_path = output_path

    def __call__(self) -> pd.DataFrame:
        orchestrator = self._build_sampler()
        samples = orchestrator.run()
        samples.to_csv(self.output_path, index=False)
        return samples

    def _build_sampler(self):

        stopping_criteria = None
        if "stopping_criteria" in self.config_dict:
            stopping_criteria = StoppingCriterionFactory.create_all_from_dict(self.config_dict["stopping_criteria"])

        orchestrator_parameters = self.config_dict.get("orchestrator_parameters", {})

        sampler = SamplerFactory(self.config).from_config(self.sampler_type, self.config_dict.get("method_parameters", {}))

        variables_types = self.config.parameters_type
        variables_values = self.config.feature_values
        if hasattr(sampler, "set_variables"):
            sampler.set_variables(variables_types, variables_values)

        if hasattr(sampler, "set_per_level_features"):
            levels = [self.config.input_parameters, self.config.design_parameters]
            sampler.set_per_level_features(levels)

        orchestrator = AdaptiveSamplingOrchestrator(
            variables_values,
            self.kernel_sampler,
            sampler,
            output_directory=self.config.output_directory,
            stopping_criteria=stopping_criteria,
            **orchestrator_parameters,
        )

        return orchestrator


class ExecutorFactory:

    def __init__(self, config: ExperimentConfig, config_dict: dict):
        self.config = config
        self.config_dict = config_dict

    def __call__(self):
        failure_resolver = self._build_failure_resolver()
        runner = self._build_runner()

        kernel_sampler = MonoKernelExecutor(runner, failure_resolver)
        return kernel_sampler

    def _build_runner(self):
        rtype = self.config_dict["runner"]
        rparam = self.config_dict.get("runner_parameters", {})

        mapping = {
            "function": self._build_function_runner,
            "executable": self._build_subprocess_runner,
        }

        if rtype not in mapping:
            raise ValueError(f"Unknown runner type '{rtype}'")

        return mapping[rtype](rparam)

    def _get_timeout(self, param):
        timeout = _get_key_or_error(
            param,
            "timeout",
            "No timeout defined for the kernel sampling module, defaulting to 30s\nSet 'timeout' to 'None' to disable timeout.",
            default=30,
            fatal=False,
        )

        if timeout == "None":
            timeout = None

        return timeout

    def _build_subprocess_runner(self, param):

        kernel = _get_key_or_error(param, "kernel", "Subprocess runner requires a 'kernel' parameter")
        kernel = _make_path_absolute(kernel, self.config.working_directory)
        objectives = self.config.objectives
        bounds = self.config.objectives_bounds
        timeout = self._get_timeout(param)
        parameters_order = _get_key_or_error(
            param,
            "parameters_order",
            "Subprocess runner requires a 'parameters_order' parameter",
        )

        res = MonoSubprocessHarness(objectives, bounds, kernel, parameters_order, timeout)
        return res

    def _build_function_runner(self, param):
        function = _get_key_or_error(param, "function", "Function runner requires a 'function' parameter")

        function = FunctionPath(function)
        # If the path is relative, then reference it to the working directory
        if function.is_source() and function.is_relative():
            function.path = _make_path_absolute(function.path, self.config.working_directory)

        objectives = self.config.objectives
        timeout = self._get_timeout(param)

        res = MonoFunctionHarness(function, objectives, timeout)
        return res

    def _build_failure_resolver(self):
        ftype = self.config_dict.get("failure_resolver", "discard")
        fparam = self.config_dict.get("failure_resolver_parameters", {})

        mapping = {
            "discard": DiscardResolver,
            "constant": ConstantResolver,
        }

        if ftype not in mapping:
            raise ValueError(f"Unknown failure resolver type '{ftype}'")

        return mapping[ftype](**fparam)


class SamplingSystemFactory:

    def __init__(self, config: ExperimentConfig, config_dict: dict):
        self.config = config
        self.config_dict = config_dict

    def __call__(self):
        kernel_sampler = ExecutorFactory(self.config, self.config_dict["SAMPLING"])()
        sampler = self._build_sampler(kernel_sampler)

        return sampler

    def _build_sampler(self, kernel_sampler):
        output_path = self.config.output_directory / "kernel_sampling/samples.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stype = self.config_dict["SAMPLING"]["sampler"]
        sparam = self.config_dict["SAMPLING"].get("sampler_parameters", {})

        mapping = {
            "ga_adaptive": _GAAdaptiveInterfaceWrapper,
            ("hvs", "multilevel_hvs"): _AdaptiveSamplerInterfaceWrapper,
            ("lhs", "random"): _StaticSamplerInterfaceWrapper,
        }

        wrapper = None
        for key in mapping:
            if isinstance(key, tuple) and stype in key:
                wrapper = mapping[key]
            elif key == stype:
                wrapper = mapping[key]

        if wrapper is None:
            raise ValueError(f"Unknown sampler type '{stype}'")

        return wrapper(kernel_sampler, self.config, stype, sparam, output_path)


def _build_kernel_sampler(config: ExperimentConfig, config_dict: dict):
    """
    Build an appropriate sampled depending on whether the sampling method is one-shot or adaptive

    Parameters
    ----------
    config
        The configuration object of the kernel sampling module

    Returns
    -------
    sampler:
        An appropriate sampler for use with the sampling method
    """

    sampler_factory = SamplingSystemFactory(config, config_dict)
    sampler = sampler_factory()

    return sampler


def sample_kernel(config: ExperimentConfig, config_dict: dict) -> pd.DataFrame:
    """
    Run the kernel sampling module on the user kernel

    Parameters
    ----------
    config
        A configuration object for the kernel sampling to execute

    Returns
    -------
    res:
        A labelled dataset of sampled points

    """
    sampler = _build_kernel_sampler(config, config_dict)

    res = sampler()
    # Ensures that the dataframe contains valid dtype
    # This is required to avoid "object" types in the dataframe (which causes issues with sklearn
    # or other models)
    res.convert_dtypes()

    return res
