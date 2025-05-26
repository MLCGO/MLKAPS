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

    def __init__(self, type):
        """
        :param type: The type of values contained (int, float, str, bool)
        :type type: type
        :rtype: None
        """
        assert type in [int, float, str, bool], f"Unknown type: {type}"
        self.type = type

    def split(self, threshold):
        """
        Split the container into two containers.
        The first container contains all values less than or equal to the threshold.
        The second container contains all values greater than the threshold.

        :param threshold: The threshold value for splitting
        :type threshold: numeric
        :rtype: tuple
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the container.
        If n_samples is greater than the number of values in the container, return all values in the container.

        :param n_samples: Number of samples to return
        :type n_samples: int
        :rtype: list or np.ndarray
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def is_continuous(self):
        """
        Return True if the container holds continuous data, False otherwise.
        Sets and Sequences for example are not continuous while a range is.

        :rtype: bool
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_size(self):
        """
        Return the size of the container.

        :rtype: int or float
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_sampling_bounds(self):
        """
        Return the lower and upper bound of the container.

        :rtype: list
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def map_to_numeric(self, data):
        """
        Map the data to float values.

        :param data: Data to map
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def map_from_numeric(self, indices):
        """
        Map numeric representation to values of the container.

        :param indices: Numeric indices to map from
        :type indices: np.ndarray or list
        :rtype: np.ndarray
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_dtype(self):
        """
        Convert a type string to a numpy dtype

        :rtype: str or type
        """
        if self.type == int:
            return "int"
        elif self.type == float:
            return "float"
        elif self.type == str:
            return str
        elif self.type == bool:
            return bool
        else:
            raise ValueError(f"Unknown variable type: {self.type}")


class ValueSet(ValueContainer):
    """Container for a set of values.
    The values in the set are not checked for anything.
    They are assumed to be valid and of the same type."""

    def __init__(self, values, type=float):
        """
        :param values: The set of values
        :type values: list or np.ndarray
        :param type: The type of values
        :type type: type
        :rtype: None
        """
        # call the parent constructor
        super().__init__(type)
        self.values = np.sort(values)

    def split(self, threshold):
        """
        Split the sequence into two sequences.
        The first sequence contains the first threshold many elements.
        The second sequence contains all values starting with index >= threshold.

        :param threshold: Index at which to split
        :type threshold: int
        :rtype: tuple
        """
        assert threshold >= 0 and threshold <= len(self.values), "Threshold must be in the index range of the set"
        threshold = round(threshold)
        return ValueSet(self.values[:threshold], self.type), ValueSet(self.values[threshold:], self.type)

    def is_continuous(self):
        """
        :rtype: bool
        """
        return False

    def get_size(self):
        """
        Return the size of the set.
        The size is defined by the number of elements in the set.

        :rtype: int
        """
        return len(self.values)

    def get_sampling_bounds(self):
        """
        Return the lower and upper bounds of the set.
        The bounds of sets are defined by their index space.

        :rtype: list
        """
        return [0, len(self.values) - 1]

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the set.
        If n_samples is greater than the number of values in the set, return all values in the set.

        :param n_samples: Number of samples to return
        :type n_samples: int
        :rtype: np.ndarray or None
        """
        if n_samples == 0:
            return None
        if n_samples < 0:
            return self.values.copy()

        n = self.get_size()
        return self.map_from_numeric(np.linspace(0, n - 1, n_samples, endpoint=True))

    def map_to_numeric(self, data):
        """
        Map the data to float values.
        The data is expected to be an array of values from the set.
        The mapping is done by creating a map of the values to their index in the set.

        :param data: Data to map
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        data = data.copy()
        feature_map = {k: j for j, k in enumerate(self.values)}
        return np.vectorize(feature_map.get)(data)

    def map_from_numeric(self, indices):
        """
        Map numeric representation to values of the container.
        Get samples by picking the values at the given indices.
        The indices are expected to be in the range of the container.
        float indices are rounded to int. Non-numeric indices are not supported.

        :param indices: Numeric indices to map from
        :type indices: np.ndarray or list
        :rtype: np.ndarray
        """
        indices = np.asarray(indices).round().astype("int")
        # creating the full sequence might not be optimal for sequences; can be improved if needed
        return self.values[indices]


class ValueSequence(ValueContainer):
    """Container for a sequence of numeric values.
    The sequence is defined by a start, stop and progression.
    It includes the start value and excludes the stop value.
    The progression mode can be "arithmetic" or "geometric"."""

    def __init__(self, start, stop, progression, mode="arithmetic", type=float):
        """
        :param start: Start value of the sequence
        :type start: int or float
        :param stop: Stop value of the sequence
        :type stop: int or float
        :param progression: Step or ratio for the sequence
        :type progression: int or float
        :param mode: Progression mode ("arithmetic" or "geometric")
        :type mode: str
        :param type: Type of the values
        :type type: type
        :rtype: None
        """
        # call the parent constructor
        super().__init__(type)
        if type not in [int, float]:
            raise ValueError(f"Unsupported type: {type}")
        if mode not in ["arithmetic", "geometric"]:
            raise ValueError(f"Unknown mode: {mode}")
        self.start = type(start)
        self.stop = type(stop)
        self.progression = type(progression)
        self.mode = mode
        assert mode != "geometric" or progression > 1, "Geometric progression must be greater than 1"

    def split(self, threshold):
        """
        Split the sequence into two sequences:
        - The first sequence contains the first elements in the sequence which are < threshold.
        - The second sequence contains all trailing values >= threshold.

        :param threshold: Value at which to split
        :type threshold: int or float
        :rtype: tuple
        """
        assert (
            threshold >= self.get_sampling_bounds()[0] and threshold <= self.get_sampling_bounds()[1] + 1
        ), "Threshold must be in the index range of the sequence"
        return (
            ValueSequence(self.start, threshold, self.progression, self.mode, self.type),
            ValueSequence(threshold, self.stop, self.progression, self.mode, self.type),
        )

    def is_continuous(self):
        """
        :rtype: bool
        """
        return False

    def get_size(self):
        """
        Return the size of the sequence.
        The size is defined by the number of elements in the sequence.

        :rtype: int
        """
        if self.mode == "arithmetic":
            return int((self.stop - self.start + self.progression - 1) // self.progression)
        assert self.mode == "geometric"
        eps = 1e-12  # np.finfo(np.float32).eps
        return int(math.log((self.stop * self.progression - eps) / self.start, self.progression))

    def get_sampling_bounds(self):
        """
        Return the lower and upper bounds of the sequence.

        :rtype: list
        """
        if self.mode == "arithmetic":
            last = self.start + self.progression * (self.get_size() - 1)
        else:
            last = self.start * (self.progression ** (self.get_size() - 1))
        return [self.start, last]

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the sequence.
        If n_samples is greater than the number of values in the sequence, return all values in the sequence.

        :param n_samples: Number of samples to return
        :type n_samples: int
        :rtype: np.ndarray or list or None
        """
        if n_samples == 0:
            return None
        if n_samples == 1:
            return [self.start]

        n = self.get_size()
        if n_samples < 0:
            n_samples = n
        indices = np.round(np.linspace(0, n - 1, n_samples, endpoint=True)).astype("int")
        if self.mode == "arithmetic":

            def gen():
                for x in range(n_samples):
                    yield self.start + indices[x] * self.progression

        else:
            # geometric
            def gen():
                for x in range(n_samples):
                    yield self.start * (self.progression ** indices[x])

        return np.fromiter(gen(), dtype=self.get_dtype(), count=n_samples)

    def map_to_numeric(self, data):
        """
        Map the data to float values.
        The data is expected to be an array of values from the set.
        The mapping is done by creating a map of the values to their index in the set.

        :param data: Data to map
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        if self.type == int:
            return np.round(data).astype("int")
        return data

    def map_from_numeric(self, data):
        """
        Map numeric representation to values of the container.
        "Quantize" input data to values in the sequence defined by start, stop and progression.

        :param data: Numeric data to map from
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        # for each element in data, find the clostest element in the sequence defined by start, stop and progression
        if self.mode == "arithmetic":
            data = np.clip(data, self.start, self.stop)
            data = np.round((data - self.start) / self.progression).astype("int")
            data = data * self.progression + self.start
        else:
            # geometric
            data = np.clip(data, *self.get_sampling_bounds())
            for i in range(len(data)):
                if data[i] > self.start:
                    # is there a more elegant way to do this?
                    pos = math.log(data[i] / self.start, self.progression)
                    low = self.start * (self.progression ** math.floor(pos))
                    high = self.start * (self.progression ** math.ceil(pos))
                    data[i] = low if data[i] - low < high - data[i] else high
        return data.astype(self.get_dtype())


class ValueRange(ValueContainer):
    """Container for a range of values.
    The range is defined by a start and stop value.
    The range is inclusive of the start value. The stop value and be either inclusive or exclusive.
    The range is always defined by floats."""

    def __init__(self, start, stop, include_high_bound=True, type=float):
        """
        :param start: Start value of the range
        :type start: float
        :param stop: Stop value of the range
        :type stop: float
        :param include_high_bound: Whether to include the upper bound
        :type include_high_bound: bool
        :param type: Type of the values (should be float)
        :type type: type
        :rtype: None
        """
        # call the parent constructor
        super().__init__(type)
        assert type == float, "Only float type is supported for continuous ranges"
        assert start <= stop, "Start must be <= stop"
        self.start = type(start)
        self.stop = type(stop)
        self.include_high_bound = include_high_bound

    def split(self, threshold):
        """
        Split the range into two ranges.
        The first range contains all values less than or equal to the threshold.
        The second range contains all values greater than the threshold.

        :param threshold: Value at which to split
        :type threshold: float
        :rtype: tuple
        """
        # is it ok that this will have threshold in both ranges?
        assert threshold >= self.start and threshold <= self.stop, "threshold must be within the range"
        return (
            ValueRange(self.start, threshold, self.include_high_bound, self.type),
            ValueRange(threshold, self.stop, self.include_high_bound, self.type),
        )

    def is_continuous(self):
        """
        :rtype: bool
        """
        return True

    def get_size(self):
        """
        Return the size of the range.
        The sizes is defined by the distance between the start and stop values.

        :rtype: float
        """
        return self.stop - self.start

    def get_sampling_bounds(self):
        """
        :rtype: list
        """
        assert self.include_high_bound, "Upper bound must be inclusive"
        return [self.start, self.stop]

    def sample_linear_space(self, n_samples):
        """
        Return a list of n_samples many values from the range.
        If n_samples is greater than the number of values in the range, return all values in the range.

        :param n_samples: Number of samples to return
        :type n_samples: int
        :rtype: np.ndarray or list
        """
        assert n_samples >= 0, "Cannot sample full continuous space, n_samples must be >= 0"
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [self.start]

        dtype = self.get_dtype()
        return np.linspace(self.start, self.stop, num=n_samples, endpoint=self.include_high_bound, dtype=dtype)

    def map_to_numeric(self, data):
        """
        Map the data to float values.
        Nothing to be done here, the data is already in the correct format.

        :param data: Data to map
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        return data.astype(self.get_dtype())

    def map_from_numeric(self, data):
        """
        Map numeric representation to values of the container.
        Nothing to be done here, the data is already in the correct format.

        :param data: Numeric data to map from
        :type data: np.ndarray or list
        :rtype: np.ndarray
        """
        return data.astype(self.get_dtype())


def _mask_variables(variables: dict, mask: list) -> dict:
    """
    Helper function to filter out variables that are not in the mask.

    :param variables: The variables to filter
    :type variables: dict
    :param mask: The list of variables to keep
    :type mask: list
    :return: Masked dictionary
    :rtype: dict
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
        :param variables_mask: A list of variables to keep. If None, all variables are kept.
        :type variables_mask: list
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
