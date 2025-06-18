import numpy as np
from mlkaps.sampling import GridSampler, ValueSet, ValueSequence, ValueRange


class TestGridSampler:
    def test_grid_sampler_set_float(self):
        v_types = {"a": "float", "b": "float"}
        a_vals = [0, 5]
        b_vals = [0, 50]
        features = {"a": ValueSet(a_vals, type=float), "b": ValueSet(b_vals, type=float)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 100, "b": 100})
        assert data.shape == (10000, 2)
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_grid_sampler_set_str(self):
        v_types = {"a": "categorical", "b": "categorical"}
        a_vals = ["0", "5"]
        b_vals = ["0", "50"]
        features = {"a": ValueSet(a_vals, type=str), "b": ValueSet(b_vals, type=str)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 100, "b": 100})
        assert data.shape == (10000, 2)
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_grid_sampler_set_int(self):
        v_types = {"a": "int", "b": "int"}
        a_vals = [0, 5, 6, 9]
        b_vals = [0, 50, 44]
        features = {"a": ValueSet(a_vals, type=int), "b": ValueSet(b_vals, type=int)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 3, "b": 2})
        assert data.shape == (6, 2)
        assert data["a"].isin(a_vals).all()
        assert data["b"].isin(b_vals).all()

    def test_grid_sampler_sequence(self):
        v_types = {"a": "float", "b": "float"}
        features = {"a": ValueSequence(2, 15, 2, "geometric"), "b": ValueSequence(0, 50, 3)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 100, "b": 100})
        assert data.shape == (10000, 2)
        assert data["a"].min() >= 2 and data["a"].max() < 15
        assert data["b"].isin(range(0, 50, 3)).all()

    def test_grid_sampler_sequence_int(self):
        v_types = {"a": "int", "b": "int"}
        features = {"a": ValueSequence(2, 15, 2, "geometric", type=int), "b": ValueSequence(0, 50, 3, type=int)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 2, "b": 15})
        assert data.shape == (2 * 15, 2)
        assert np.issubdtype(data["a"].values.dtype, np.integer)
        assert np.issubdtype(data["b"].values.dtype, np.integer)
        assert data["a"].min() >= 2 and data["a"].max() < 15
        assert data["b"].isin(range(0, 50, 3)).all()

    def test_grid_sampler_range(self):
        v_types = {"a": "float", "b": "float"}
        features = {"a": ValueRange(2, 15), "b": ValueRange(0, 50)}
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 100, "b": 100})
        assert data.shape == (100 * 100, 2)

    def test_grid_sampler_mixed(self):
        v_types = {"a": "int", "b": "float", "c": "categorical"}
        c_vals = ["0", "5", ".d"]
        features = {
            "a": ValueSequence(2, 15, 2, "geometric", type=int),
            "b": ValueSequence(0, 50, 3, type=float),
            "c": ValueSet(c_vals),
        }
        sampler = GridSampler(v_types, features)
        data = sampler.sample({"a": 2, "b": 10, "c": 3})
        assert data.shape == (2 * 10 * 3, 3)
        assert np.issubdtype(data["a"].values.dtype, "int")
        assert data["b"].values.dtype == "float"
        assert data["c"].values.dtype == "object"
        assert data["a"].min() >= 2 and data["a"].max() < 15
        assert data["b"].isin(range(0, 50, 3)).all()
        assert data["c"].isin(c_vals).all()
