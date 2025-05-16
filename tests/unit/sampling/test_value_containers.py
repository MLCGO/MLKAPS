import numpy as np
from mlkaps.sampling import ValueSet, ValueSequence, ValueRange


class TestValueContainers:
    def test_value_set(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values)
        assert value_set.sample_linear_space(3) == [1, 2, 3]

    def test_value_set_unsorted(self):
        values = [5, 1, 3, 4, 2]
        value_set = ValueSet(values)
        assert value_set.sample_linear_space(3) == [1, 2, 3]

    def test_value_set_str(self):
        values = ["x", "2", ".d", "a", "glk"]
        value_set = ValueSet(values)
        assert value_set.sample_linear_space(3) == [".d", "2", "a"]

    def test_value_set_sample(self):
        values = [1, 2, 3, 4, 5]
        value_set = ValueSet(values)
        assert np.array_equal(value_set.get_samples([0, 1, 2]), [1, 2, 3])
        assert np.array_equal(value_set.get_samples([0, 1]), [1, 2])
        assert np.array_equal(value_set.get_samples([0]), [1])
        assert np.array_equal(value_set.get_samples([4, 1, 2, 3]), [5, 2, 3, 4])

    def test_value_set_sample_str(self):
        values = ["x", "2", ".d", "a", "glk"]
        value_set = ValueSet(values)
        assert np.array_equal(value_set.get_samples([0, 1, 2]), [".d", "2", "a"])
        assert np.array_equal(value_set.get_samples([0, 1]), [".d", "2"])
        assert np.array_equal(value_set.get_samples([0]), [".d"])
        assert np.array_equal(value_set.get_samples([4, 1, 2, 3]), ["x", "2", "a", "glk"])

    def test_value_sequence_arithmetic_int(self):
        value_sequence = ValueSequence(1, 20, 3, "arithmetic")
        assert np.array_equal(value_sequence.sample_linear_space(5, "int"), [1, 4, 7, 10, 13])
        assert np.array_equal(value_sequence.sample_linear_space(10, "int"), [1, 4, 7, 10, 13, 16, 19])

    def test_value_sequence_arithmetic_float(self):
        value_sequence = ValueSequence(1.0, 20.0, 3.0, "arithmetic")
        assert np.allclose(value_sequence.sample_linear_space(5, "float"), [1.0, 4.0, 7.0, 10.0, 13.0], rtol=1e-05)
        assert np.allclose(
            value_sequence.sample_linear_space(10, "float"), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0], rtol=1e-05
        )

    def test_value_sequence_geometric_int(self):
        value_sequence = ValueSequence(2, 129, 2, "geometric")
        assert np.array_equal(value_sequence.sample_linear_space(3, "int"), [2, 4, 8])
        assert np.array_equal(value_sequence.sample_linear_space(10, "int"), [2, 4, 8, 16, 32, 64, 128])
        value_sequence = ValueSequence(2, 128, 2, "geometric")
        assert np.array_equal(value_sequence.sample_linear_space(10, "int"), [2, 4, 8, 16, 32, 64])

    def test_value_sequence_geometric_float(self):
        value_sequence = ValueSequence(2.0, 129.0, 2.0, "geometric")
        assert np.allclose(value_sequence.sample_linear_space(3, "float"), [2.0, 4.0, 8.0], rtol=1e-05)
        assert np.allclose(
            value_sequence.sample_linear_space(10, "float"), [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], rtol=1e-05
        )
        value_sequence = ValueSequence(2.0, 128.0, 2.0, "geometric")
        assert np.allclose(value_sequence.sample_linear_space(10, "float"), [2.0, 4.0, 8.0, 16.0, 32.0, 64.0], rtol=1e-05)

    def test_value_sequence_sample(self):
        value_sequence = ValueSequence(1, 20, 3, "arithmetic")
        assert np.array_equal(value_sequence.get_samples([0, 1, 2]), [1, 4, 7])
        assert np.array_equal(value_sequence.get_samples([0, 1]), [1, 4])
        assert np.array_equal(value_sequence.get_samples([0]), [1])
        assert np.array_equal(value_sequence.get_samples([4, 1, 2, 3]), [13, 4, 7, 10])

    def tests_value_range(self):
        value_range = ValueRange(0, 20)
        assert np.array_equal(value_range.sample_linear_space(5), [0, 5, 10, 15, 20])
        value_range = ValueRange(0, 20, False)
        assert np.array_equal(value_range.sample_linear_space(5), [0, 4, 8, 12, 16])
