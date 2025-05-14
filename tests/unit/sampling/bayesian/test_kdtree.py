
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
        
    def test_can_compute_size(self):
        bounds = {"x": [0, 2], "y": [0, 2]}
        tree = KDTree(bounds, {"x": "float", "y": "float"})

        assert tree.nodes[0].volume == 4

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

        # speed-test
        data = pd.DataFrame({"x": np.random.randint(0, 2, 1000000), "y": np.random.randint(0, 2, 1000000)})

        import time
        time.start = time.time()
        nrepet = 500
        for i in range(nrepet):
            result = tree.predict(data)
        time.end = time.time()
        print(f"Speed test: {(time.end - time.start) / nrepet} seconds")