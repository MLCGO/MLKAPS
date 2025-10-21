from mlkaps.sampling.bayesian.random_bayesian import RandomBayesianSampler
from mlkaps.sampling import ValueRange


class TestRandomBayesianSampler:

    @staticmethod
    def d2_synth_kernel(x):
        return x["x1"] ** 2 + x["x2"] ** 2

    @staticmethod
    def harness_kernel(x):
        d = x.copy()
        d["y"] = TestRandomBayesianSampler.d2_synth_kernel(x)
        return d

    def test_can_build(self):
        features = {"x1": ValueRange(0, 1), "x2": ValueRange(0, 1)}
        feature_types = {"x1": "float", "x2": "float"}
        directions = {"y": "minimize"}

        sampler = RandomBayesianSampler(TestRandomBayesianSampler.harness_kernel, ["x1"], features, feature_types, directions)
        assert sampler.input_features == ["x1"]
        assert sampler.design_parameters == ["x2"]
        for feature in features:
            assert sampler.feature_values[feature].get_sampling_bounds() == features[feature].get_sampling_bounds()

    def test_can_sample(self):
        features = {"x1": ValueRange(0, 1), "x2": ValueRange(0, 1)}
        feature_types = {"x1": "float", "x2": "float"}
        directions = {"y": "minimize"}

        sampler = RandomBayesianSampler(
            TestRandomBayesianSampler.harness_kernel, ["x1"], features, feature_types, directions, bootstrap_ratio=0.14
        )
        sampler(None, 100)
