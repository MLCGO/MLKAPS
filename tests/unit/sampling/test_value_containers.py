import numpy as np
from mlkaps.sampling import ValueSet, ValueSequence, ValueRange


class TestValueContainers:
    def test_value_set(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values, int)
        assert np.array_equal(value_set.sample_linear_space(3), [1, 3, 5])
        assert np.array_equal(value_set.sample_linear_space(), values)
        assert np.array_equal(value_set.get_sampling_bounds(), (0, 4))

    def test_value_set_unsorted(self):
        values = [5, 1, 3, 4, 2]
        value_set = ValueSet(values, int)
        assert np.array_equal(value_set.sample_linear_space(3), [1, 3, 5])
        assert np.array_equal(value_set.sample_linear_space(), sorted(values))
        assert np.array_equal(value_set.get_sampling_bounds(), (0, 4))

    def test_value_set_str(self):
        values = ["x", "2", ".d", "a", "glk"]
        value_set = ValueSet(values, str)
        assert np.array_equal(value_set.sample_linear_space(3), [".d", "a", "x"])
        assert np.array_equal(value_set.sample_linear_space(), sorted(values))

    def test_value_set_sample(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values, float)
        assert np.array_equal(value_set.map_from_numeric([0, 1, 2]), [1, 2, 3])
        assert np.array_equal(value_set.map_from_numeric([0, 1]), [1, 2])
        assert np.array_equal(value_set.map_from_numeric([0]), [1])
        assert np.array_equal(value_set.map_from_numeric([4.4, 1.1, 1.9, 3.4]), [5, 2, 3, 4])

    def test_value_set_sample_str(self):
        values = ["x", "2", ".d", "a", "glk"]
        value_set = ValueSet(values)
        assert np.array_equal(value_set.map_from_numeric([0, 1, 2]), [".d", "2", "a"])
        assert np.array_equal(value_set.map_from_numeric([0, 1]), [".d", "2"])
        assert np.array_equal(value_set.map_from_numeric([0.0]), [".d"])
        assert np.array_equal(value_set.map_from_numeric([4.1, 1.1, 1.9, 3.4]), ["x", "2", "a", "glk"])

    def test_value_sequence_arithmetic_int(self):
        value_sequence = ValueSequence(1, 20, 3, "arithmetic", type=int)
        assert np.array_equal(value_sequence.sample_linear_space(5), [1, 7, 10, 13, 19])
        assert np.array_equal(value_sequence.sample_linear_space(10), [1, 4, 4, 7, 10, 10, 13, 16, 16, 19])
        assert np.array_equal(value_sequence.get_sampling_bounds(), (1, 19))

    def test_value_sequence_arithmetic_float(self):
        value_sequence = ValueSequence(1.0, 20, 3.0, "arithmetic", type=float)
        assert np.allclose(value_sequence.sample_linear_space(5), [1, 7, 10, 13, 19], rtol=1e-05)
        assert np.allclose(value_sequence.sample_linear_space(10), [1, 4, 4, 7, 10, 10, 13, 16, 16, 19], rtol=1e-05)
        assert np.allclose(value_sequence.sample_linear_space(), [1, 4, 7, 10, 13, 16, 19], rtol=1e-05)
        assert np.array_equal(value_sequence.get_sampling_bounds(), (1, 19))

    def test_value_sequence_geometric_int(self):
        value_sequence = ValueSequence(2, 129, 2, "geometric", type=int)
        assert np.array_equal(value_sequence.sample_linear_space(3), [2, 16, 128])
        assert np.array_equal(value_sequence.sample_linear_space(10), [2, 4, 4, 8, 16, 16, 32, 64, 64, 128])
        assert np.array_equal(value_sequence.sample_linear_space(), [2, 4, 8, 16, 32, 64, 128])
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 128))
        value_sequence = ValueSequence(2, 128, 2, "geometric", type=int)
        assert np.array_equal(value_sequence.sample_linear_space(10), [2, 4, 4, 8, 8, 16, 16, 32, 32, 64])
        assert np.array_equal(value_sequence.sample_linear_space(), [2, 4, 8, 16, 32, 64])
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 64))

    def test_value_sequence_geometric_float(self):
        value_sequence = ValueSequence(2.0, 129.0, 2.0, "geometric", type=float)
        assert np.allclose(value_sequence.sample_linear_space(3), [2.0, 16.0, 128.0], rtol=1e-05)
        assert np.allclose(value_sequence.sample_linear_space(10), [2, 4, 4, 8, 16, 16, 32, 64, 64, 128], rtol=1e-05)
        assert np.array_equal(value_sequence.sample_linear_space(), [2, 4, 8, 16, 32, 64, 128])
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 128))
        value_sequence = ValueSequence(2.0, 128.0, 2.0, "geometric", type=float)
        assert np.allclose(value_sequence.sample_linear_space(10), [2, 4, 4, 8, 8, 16, 16, 32, 32, 64], rtol=1e-05)
        assert np.array_equal(value_sequence.sample_linear_space(), [2, 4, 8, 16, 32, 64])
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 64))
        value_sequence = ValueSequence(2.0, 128.0001, 2.0, "geometric", type=float)
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 128))
        value_sequence = ValueSequence(2.0, 128.0001, 2.0, "geometric", type=float)
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 128))
        value_sequence = ValueSequence(2.0, 162, 3.0, "geometric", type=float)
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 54))
        value_sequence = ValueSequence(2.0, 162.001, 3.0, "geometric", type=float)
        assert np.array_equal(value_sequence.get_sampling_bounds(), (2, 162))

    def test_value_sequence_map_from_numeric_int(self):
        value_sequence = ValueSequence(1, 20, 3, "arithmetic", type=int)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1, 2, 3, 4]), [1, 1, 1, 4, 4])
        assert np.array_equal(value_sequence.map_from_numeric([4, 1, 2, 9]), [4, 1, 1, 10])
        value_sequence = ValueSequence(2, 129, 2, "geometric", type=int)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1, 2, 3, 4]), [2, 2, 2, 4, 4])
        assert np.array_equal(value_sequence.map_from_numeric([4, 1, 2, 9]), [4, 2, 2, 8])
        value_sequence = ValueSequence(2, 129, 3, "geometric", type=int)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1, 2, 3, 4]), [2, 2, 2, 2, 6])
        assert np.array_equal(value_sequence.map_from_numeric([4, 1, 2, 13]), [6, 2, 2, 18])

    def test_value_sequence_map_from_numeric_float(self):
        value_sequence = ValueSequence(1.0, 20.0, 3.0, "arithmetic", type=float)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1, 2, 3, 4]), [1, 1, 1, 4, 4])
        assert np.array_equal(value_sequence.map_from_numeric([4.4, 1.1, 1.9, 9.3]), [4, 1, 1, 10])
        value_sequence = ValueSequence(2.0, 129.0, 2.0, "geometric", type=float)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1.1, 2, 3, 4.001]), [2, 2, 2, 4, 4])
        assert np.array_equal(value_sequence.map_from_numeric([4.1, 1.1, 1.9, 8.9]), [4, 2, 2, 8])
        value_sequence = ValueSequence(2, 129, 3, "geometric", type=float)
        assert np.array_equal(value_sequence.map_from_numeric([0, 1.1, 3.99, 4.0, 4.01]), [2, 2, 2, 6, 6])
        assert np.array_equal(value_sequence.map_from_numeric([4.5, 1, 2, 13]), [6, 2, 2, 18])

    def tests_value_range(self):
        value_range = ValueRange(0, 20)
        assert np.array_equal(value_range.sample_linear_space(5), [0, 5, 10, 15, 20])
        value_range = ValueRange(0, 20, False)
        assert np.array_equal(value_range.sample_linear_space(5), [0, 4, 8, 12, 16])

    def test_split_set(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values)
        splits = value_set.split(2)
        assert np.array_equal(splits[0].sample_linear_space(), [1, 2])
        assert np.array_equal(splits[1].sample_linear_space(), [3, 4, 5])
        splits = value_set.split(5)
        assert np.array_equal(splits[0].sample_linear_space(), [1, 2, 3, 4, 5])
        assert np.array_equal(splits[1].sample_linear_space(), [])
        splits = value_set.split(0)
        assert np.array_equal(splits[1].sample_linear_space(), [1, 2, 3, 4, 5])
        assert np.array_equal(splits[0].sample_linear_space(), [])

    def test_split_sequence_arithmetic(self):
        value_seq = ValueSequence(1, 6, 1, "arithmetic", type=int)
        splits = value_seq.split(3)
        assert np.array_equal(splits[0].sample_linear_space(), [1, 2])
        assert np.array_equal(splits[1].sample_linear_space(), [3, 4, 5])
        splits = value_seq.split(6)
        assert np.array_equal(splits[0].sample_linear_space(), [1, 2, 3, 4, 5])
        assert np.array_equal(splits[1].sample_linear_space(), [])
        splits = value_seq.split(1)
        assert np.array_equal(splits[0].sample_linear_space(), [])
        assert np.array_equal(splits[1].sample_linear_space(), [1, 2, 3, 4, 5])

    def test_split_sequence_geometric(self):
        value_seq = ValueSequence(2.0, 129.0, 2.0, "geometric", type=float)
        splits = value_seq.split(64)
        assert np.array_equal(splits[0].sample_linear_space(), [2, 4, 8, 16, 32])
        assert np.array_equal(splits[1].sample_linear_space(), [64, 128])

    def test_split_range(self):
        value_range = ValueRange(1, 5)
        splits = value_range.split(2)
        assert np.array_equal(splits[0].sample_linear_space(2), [1, 2])
        assert np.array_equal(splits[1].sample_linear_space(3), [2.0, 3.5, 5.0])

    def test_get_size_set(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values)
        assert value_set.get_size() == 5

    def test_get_size_sequence(self):
        value_sequence = ValueSequence(1, 20, 3, "arithmetic", type=int)
        assert value_sequence.get_size() == 7
        value_sequence = ValueSequence(2.0, 200.0, 3.0, "geometric")
        assert value_sequence.get_size() == 5
        value_sequence = ValueSequence(20, 20, 3.0, "geometric", type=float)
        assert value_sequence.get_size() == 0
        value_sequence = ValueSequence(200.0, 200.0001, 3.0, "geometric", type=float)
        assert value_sequence.get_size() == 1

    def test_get_size_range(self):
        value_range = ValueRange(1, 20)
        assert value_range.get_size() == 19.0
