"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Define the base class for all sampler
"""

import numpy as np
import math


class ValueContainer:
    """Base class for value containers.
    Value containers are used, for example, to store parameters and by samplers."""

    def sample_linear_space(self, n_samples=-1, type="float"):
        """Return a list of n_samples many values from the container.
        If n_samples is greater than the number of values in the container, return all values in the container."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def is_continuous(self):
        """Return True if the container holds continuous data, False otherwise.
        Sets and Sequences for example are not continuous while a range is."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_lower_sampling_bound(self):
        """Return the lower bound of the container."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_upper_sampling_bound(self):
        """Return the upper bound of the container and an indication of the bound is inclusive or not.
        The return value is a tuple (upper_bound, is_inclusive)."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_samples(self, indices, type="float"):
        """Get samples by picking the values at the given indices.
        The indices are expected to be in the range of the container.
        float indices are rounded to int. Non-numeric indices are not supported."""
        indices = np.asarray(indices).round().astype("int")
        # creating the full sequence might not be optimal for sequences; can be improved if needed
        vals = np.asarray(self.sample_linear_space())
        return vals[indices]

    def get_dtype(self, type):
        "Convert a type string to a numpy dtype"
        if type == "int":
            return "int"
        elif type == "float":
            return "float"
        elif type == "Categorical":
            return str
        elif type == "Boolean":
            return bool
        else:
            raise ValueError(f"Unknown variable type: {type}")


class ValueSet(ValueContainer):
    """Container for a set of values.
    The values in the set are not checked for anything. They are assumed to be valid and of the same type."""

    def __init__(self, values):
        self.values = sorted(values)

    def is_continuous(self):
        return False

    def get_lower_sampling_bound(self):
        return 0

    def get_upper_sampling_bound(self):
        return len(self.values) - 1

    def sample_linear_space(self, n_samples=-1, type="float"):
        """Return a list of n_samples many values from the set.
        If n_samples is greater than the number of values in the set, return all values in the set."""
        if n_samples == 0:
            return None
        if n_samples < 0:
            n_samples = len(self.values)
        return self.values[0:n_samples] if n_samples < len(self.values) else self.values


class ValueSequence(ValueContainer):
    """Container for a sequence of values.
    The sequence is defined by a start, stop and progression. It includes the start value and excludes the stop value.
    The progression mode can be "arithmetic" or "geometric"."""

    def __init__(self, start, stop, progression, mode="arithmetic"):
        if mode not in ["arithmetic", "geometric"]:
            raise ValueError(f"Unknown mode: {mode}")
        self.start = start
        self.stop = stop
        self.progression = progression
        self.mode = mode
        assert mode != "geometric" or progression > 1, "Geometric progression must be greater than 1"

    def is_continuous(self):
        return False

    def get_lower_sampling_bound(self):
        return 0

    def get_upper_sampling_bound(self):
        if self.mode == "arithmetic":
            return (self.stop - self.start + self.progression - 1) // self.progression - 1
        assert self.mode == "geometric"
        return int(math.log(self.stop - self.start + self.progression - 1, self.progression)) - 1

    def sample_linear_space(self, n_samples=-1, type="float"):
        """Return a list of n_samples many values from the sequence.
        If n_samples is greater than the number of values in the sequence, return all values in the sequence."""
        if type not in ["int", "float"]:
            raise ValueError(f"Unknown type: {type}")
        if n_samples == 0:
            return None
        if n_samples == 1:
            return [self.start]

        dtype = self.get_dtype(type)
        if self.mode == "arithmetic":
            stop = self.stop
            if n_samples > 0:
                stop = min(self.stop, 1 + self.start + self.progression * (n_samples - 1))
            return np.arange(self.start, stop, self.progression, dtype=dtype)

        assert self.mode == "geometric"
        n = int(math.log(self.stop - self.start + self.progression - 1, self.progression))
        if n_samples > 0:
            n = min(n_samples, n)
        res = np.full(n, self.progression, dtype=dtype)
        res[0] = self.start
        return np.cumprod(res)


class ValueRange(ValueContainer):
    """Container for a range of values.
    The range is defined by a start and stop value.
    The range is inclusive of the start value. The stop value and be either inclusive or exclusive.
    The range is always defined by floats."""

    def __init__(self, start, stop, include_high_bound=True):
        self.start = start
        self.stop = stop
        self.include_high_bound = include_high_bound

    def is_continuous(self):
        return True

    def get_lower_sampling_bound(self):
        return self.start

    def get_upper_sampling_bound(self):
        assert self.include_high_bound, "Upper bound must be inclusive"
        return self.stop

    def sample_linear_space(self, n_samples, type="float"):
        """Return a list of n_samples many values from the range.
        If n_samples is greater than the number of values in the range, return all values in the range."""
        assert n_samples >= 0, "Cannot sample full continuous space, n_samples must be >= 0"
        assert type == "float", "Only float type is supported for continuous ranges"
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [self.start]

        dtype = self.get_dtype(type)
        return np.linspace(self.start, self.stop, num=n_samples, endpoint=self.include_high_bound, dtype=dtype)

    def get_samples(self, indices, type="float"):
        raise ValueError("Cannot index continuous space")


def _mask_variables(variables: dict, mask: list) -> dict:
    """
    Helper function to filter out variables that are not in the mask.


    :param variables: The variables to filter
    :type variables: dict
    :param mask: The list of variables to keep
    :type mask: list


    :return: Masked dictionary
    :dict: dict
    """
    if mask is None:
        return variables
    return {key: value for key, value in variables.items() if key in mask}


class SamplerError(Exception):
    """
    Generic exception to raise when a sampler fails
    """


class Sampler:
    """
    Base class for all samplers.
    """

    def __init__(self, variables_types=None, variables_values=None, variables_mask=None):
        """
        Initializes the sampler.

        :param variables_types: A dictionary associating the name of each variable to its type. The type can be either
            ["Categorical", "Boolean", "int", "float"]. If None, then the variables types must be
            set using the set_variables method before sampling.
        :type variables_types:  dict
        :param variables_values:
            A dictionary associating the name of each variable to its possible values.
            Continuous variables (int, float) must be a range [min, max]
            Categorical/Boolean variables must be a list of possible values
        :type variables_values:  dict
        """
        self.variables_values = None
        self.variables_types = None

        # Set the variables using a setter to ensure that overriding classes can perform
        # additional checks if needed
        self.set_variables(variables_types, variables_values, variables_mask)

    def _raise_if_variables_not_set(self):
        """
        :raise SamplerError: Raise an exception if the variables are not set,
        or if the variables are empty (after masking for example).
        """

        if self.variables_values is None or self.variables_types is None:
            raise SamplerError("The sampler variables (values and/or types) were not set!")
        if len(self.variables_values) == 0 or len(self.variables_types) == 0:
            raise SamplerError("The passed variables were empty, or all variables were masked out!")

    def set_variables(self, variables_types: dict, variables_values: dict, mask: list = None):
        """
        Sets the variables to be sampled.

        :param variables_types:
            A dictionary associating the name of each variable to its type. The type can be either
            ["Categorical", "Boolean", "int", "float"].
            If none, then the variables are cleared and must be set again before sampling.
        :type variables_types: dict
        :param variables_values:
            A dictionary associating the name of each variable to its possible values. The possible
            values must be a list of values for categorical variables, or a tuple (min, max) for
            numerical variables.
            If none, then the variables are cleared and must be set again before sampling.
        :type variables_values: dict
        :param mask:
            A list of variables to keep. If None, all variables are kept. This can be useful when
            the variables to be samples are a subset of the variables defined in the sampler.
        :type mask: list
        """

        # Filter out masked variables
        variables_values = _mask_variables(variables_values, mask)
        self.variables_values = variables_values

        variables_types = _mask_variables(variables_types, mask)
        self.variables_types = variables_types

        if variables_values is None or variables_types is None:
            return

        # Ensure that both dict contain the same keys (variables)
        assert sorted(self.variables_types.keys()) == sorted(self.variables_values.keys())
