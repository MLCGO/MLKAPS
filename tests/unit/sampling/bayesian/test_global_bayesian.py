from mlkaps.sampling.bayesian.global_bayesian import BayesianSampler, KDTreeSplitter
from mlkaps.sampling.bayesian.kdtree import KDTree, KDTreeNode
from mlkaps.modeling.quantile_variance_estimator import QuantileLGBMMixture
import numpy as np
import pandas as pd
import copy

class TestBayesianSplitter:

    def test_can_build(self):
        objective = "y"
        directions = {"y": "minimize"}
        feature_types = {"x1": "float", "x2": "float"}
        features = {"x1": [0, 1], "x2": [0, 1]}

        partitionner = KDTree({k: v for k, v in features.items() if k in ["x1"]}, feature_types)
        model = QuantileLGBMMixture()

        KDTreeSplitter(objective, directions, partitionner, model)

    def test_can_variance_split(self):
        objective = "y"
        directions = {"y": "minimize"}
        feature_types = {"x1": "float", "x2": "float"}
        features = {"x1": [0, 1], "x2": [0, 1]}

        partitionner = KDTree({k: v for k, v in features.items() if k in ["x1"]}, feature_types)
        model = QuantileLGBMMixture()

        random_data = pd.DataFrame({"x1": np.random.rand(100), "x2": np.random.rand(100)})
        random_data["y"] = random_data.apply(lambda x: 1 if x["x1"] < 0.5 else 5, axis=1)

        splitter = KDTreeSplitter(objective, directions, partitionner, model)
        splitter.variance_split(random_data, 1)
        assert 0.48 < partitionner.thresholds[0] < 0.52

        # We perform a variance split, but not from the root
        random_data["x1"] /= 2
        splitter.variance_split(random_data, 1, 1)
        assert 0.23 < partitionner.thresholds[1] < 0.27

    def test_can_follow_directions(self):
        objective = "y"
        directions = {"y": "minimize"}
        feature_types = {"x1": "float", "x2": "float"}
        features = {"x1": [0, 1], "x2": [0, 1]}

        partitionner = KDTree({k: v for k, v in features.items() if k in ["x1"]}, feature_types)
        model = QuantileLGBMMixture()

        splitter = KDTreeSplitter(objective, directions, partitionner, model)

        data = pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.1, 0.2], "y": [0.1, 0.2]})

        assert splitter._is_better(0.1, 0.5) == True
        assert splitter._is_better(0.5, 0.1) == False
        assert splitter._idx_of_optimum(data) == 0

        splitter = KDTreeSplitter(objective, {"y": "maximize"}, partitionner, model)
        assert splitter._is_better(0.1, 0.5) == False
        assert splitter._is_better(0.5, 0.1) == True
        assert splitter._idx_of_optimum(data) == 1

        
        




class TestBayesianSampler:

    @staticmethod
    def d2_synth_kernel(x):
        return x["x1"] ** 2 + x["x2"] ** 2

    @staticmethod
    def harness_kernel(x):
        d = x.copy()
        d["y"] = TestBayesianSampler.d2_synth_kernel(x)
        return d

    def test_can_build(self):
        features = {"x1": [0, 1], "x2": [0, 1]}
        feature_types = {"x1": "float", "x2": "float"}
        directions = {"y": "minimize"}

        sampler = BayesianSampler(TestBayesianSampler.harness_kernel, ["x1"], features, feature_types, directions, "EI_size")
        assert sampler.input_features == ["x1"]
        assert sampler.design_parameters == ["x2"]
        assert sampler.feature_values == features

    def test_can_sample(self):
        features = {"x1": [0, 1], "x2": [0, 1]}
        feature_types = {"x1": "float", "x2": "float"}
        directions = {"y": "minimize"}

        sampler = BayesianSampler(
            TestBayesianSampler.harness_kernel, ["x1"], features, feature_types, directions, "EI_size", bootstrap_ratio=0.14
        )
        sampler(None, 100)
