
from mlkaps.sampling.bayesian.kdtree import KDTree, KDTreeNode
import pytest
import numpy as np
import pandas as pd

class TestKDTree:

    def test_can_create(self):
        bounds = {"x": [0, 1], "y": [0, 1]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        assert isinstance(tree.nodes[0], KDTreeNode)
        assert tree.nodes[0].bounds == bounds

    def test_can_split(self):
        bounds = {"x": [0, 1], "y": [0, 1]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        tree.split(0, "x", 0.5)
        assert len(tree.nodes) == 3

        # Test wether the bounds are correctly set
        assert tree.nodes[0].bounds == {"x": [0, 1], "y": [0, 1]}
        assert tree.nodes[1].bounds == {"x": [0.0, 0.5], "y": [0, 1]}
        assert tree.nodes[2].bounds == {"x": [0.5, 1], "y": [0, 1]}

        # Test whether the parent and depth are correctly set
        assert tree.nodes[1].parent == 0
        assert tree.nodes[2].parent == 0

        assert tree.nodes[1].depth == 1
        assert tree.nodes[2].depth == 1

        # Test whether the split axis and threshold are correctly set
        assert tree.thresholds[0] == 0.5
        assert tree.split_axis[0] == 0

        # Test whether the children are correctly set
        assert tree.thresholds[1] == -1 and tree.split_axis[1] == -1
        assert tree.thresholds[2] == -1 and tree.split_axis[2] == -1
        assert tree.lefts[1] == -1 and tree.rights[1] == -1
        assert tree.lefts[2] == -1 and tree.rights[2] == -1

        # Test whether the left and right children are correctly set
        assert tree.lefts[0] == 1
        assert tree.rights[0] == 2

    def test_raise_invalid_split(self):
        bounds = {"x": [0, 1], "y": [0, 1]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        with pytest.raises(ValueError):
            tree.split(0, "z", 0.5)

        with pytest.raises(ValueError):
            tree.split(0, "x", 2)

        with pytest.raises(ValueError):
            tree.split(0, "y", -1)

        # Test raises an error when double splitting a node
        tree.split(0, "x", 0.5)
        with pytest.raises(ValueError):
            tree.split(0, "x", 0.5)
        
        with pytest.raises(IndexError):
            tree.split(-5, "y", 0.5)

        with pytest.raises(IndexError):
            tree.split(8, "x", 0.5)


    def test_can_compute_volume(self):
        bounds = {"x": [0, 2], "y": [0, 2]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        assert tree.nodes[0].volume == 4
        tree.split(0, "x", 1)

        assert tree.nodes[0].volume == 4
        assert tree.nodes[1].volume == 2
        assert tree.nodes[2].volume == 2

    def test_can_predict(self):
        bounds = {"x": [0, 2], "y": [0, 2]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        childs = tree.nodes[0].split("x", 1)
        childs[1].split("y", 1)

        data = pd.DataFrame({"x": [0, 0.5, 1.5, 2], "y": [0, 1.5, 0.5, 2]})
        result = tree.predict(data)

        assert len(result) == len(data)
        assert np.all(result >= 0) and np.all(result < len(tree.nodes))
        assert np.all(result == [1, 1, 3, 4])

    def test_raise_invalid_predict(self):
        bounds = {"x": [0, 2], "y": [0, 2]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        data = pd.DataFrame({"x": [0, 0.5, 1.5, 2]})
        with pytest.raises(Exception):
            tree.predict(data[["x"]])