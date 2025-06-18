import numpy as np
import pytest
from mlkaps.sampling import ValueSet, ValueSequence, ValueRange
from mlkaps.sampling.generic_bounded_sampler import LhsSampler, RandomSampler


@pytest.mark.parametrize("sampler", [LhsSampler, RandomSampler])
class TestGenericBoundedSampler:
    def test_lhs_sampler_set_float(self, sampler):
        v_types = {"a": "float", "b": "float"}
        a_vals = [0.1, 5.2, 7.4]
        b_vals = [0.3, 50.4, 200.5]
        features = {"a": ValueSet(a_vals), "b": ValueSet(b_vals)}
        thesampler = sampler(v_types, features)
        data = thesampler.sample(10)
        assert data.shape == (10, 2)
        assert data["a"].values.dtype == "float"
        assert data["b"].values.dtype == "float"
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_lhs_sampler_set_str(self, sampler):
        v_types = {"a": "categorical", "b": "categorical"}
        a_vals = ["hx", "xg"]
        b_vals = ["gz", "zg"]
        features = {"a": ValueSet(a_vals), "b": ValueSet(b_vals)}
        thesampler = sampler(v_types, features)
        data = thesampler.sample(10)
        assert data.shape == (10, 2)
        assert data["a"].values.dtype == "object"
        assert data["b"].values.dtype == "object"
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_lhs_sampler_set_int(self, sampler):
        v_types = {"a": "int", "b": "int"}
        a_vals = [0, 5, 7]
        b_vals = [1, 50, 200]
        features = {"a": ValueSet(a_vals), "b": ValueSet(b_vals)}
        thesampler = sampler(v_types, features)
        data = thesampler.sample(10)
        assert data.shape == (10, 2)
        assert np.issubdtype(data["a"].values.dtype, np.integer)
        assert np.issubdtype(data["b"].values.dtype, np.integer)
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_lhs_sampler_sequence(self, sampler):
        v_types = {"a": "float", "b": "float"}
        features = {"a": ValueSequence(2, 15, 2, "geometric"), "b": ValueSequence(0.0, 50.0, 3.0)}
        thesampler = sampler(v_types, features)
        data = thesampler.sample(10)
        assert data.shape == (10, 2)
        assert data["a"].values.dtype == "float"
        assert data["b"].values.dtype == "float"
        assert data["a"].min() >= 2 and data["a"].max() < 15
        assert data["b"].isin(range(0, 50, 3)).all()

    def test_lhs_sampler_range(self, sampler):
        v_types = {"a": "float", "b": "float"}
        features = {"a": ValueRange(2, 15), "b": ValueRange(0, 50)}
        thesampler = sampler(v_types, features)
        data = thesampler.sample(17)
        assert data.shape == (17, 2)
        assert data["a"].values.dtype == "float"
        assert data["b"].values.dtype == "float"
        assert data["a"].min() >= 2.0 and data["a"].max() <= 15.0
        assert data["b"].min() >= 0.0 and data["b"].max() <= 50.0
