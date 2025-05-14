"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
import pandas as pd

try:
    import numba
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    import warnings
    warnings.warn("Numba is not available. JIT compilation will be disabled.", ImportWarning)

def maybe_jit(*args, **kwargs):
    def decorator(func):
        return numba.jit(*args, **kwargs)(func) if JIT_AVAILABLE else func
    return decorator

class KDTreeNode:
    __slots__ = ['bounds', 'id', 'parent', 'left', 'right', 'threshold', 'split_axis', 'tree', 'data']

    def __init__(self, bounds: dict[str, list[float]], id, parent=None, tree=None):
        self.bounds = bounds
        
        self.id = id
        assert parent or tree, "Either parent or tree must be provided"

        self.parent = parent
        self.tree = tree if tree else parent.tree
        
        self.data = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None
    
    @property
    def volume(self):
        size = 1
        for low, high in self.bounds.values():
            size *= (high - low)
        return size
    
    def split(self, axis, threshold):
        return self.tree.split(self.id, threshold, axis)

@maybe_jit(nogil=True)
def _tree_predict(X, lefts, rights, split_axes, thresholds):
    indices = np.arange(X.shape[0])
    stack = [(0, indices)]
    res = np.empty(X.shape[0], dtype=np.int64)
    while len(stack) != 0:
        curr_id, indices = stack.pop()

        if lefts[curr_id] == -1 and rights[curr_id] == -1:
            res[indices] = curr_id
            continue

        data = X[indices]
        left = data[:, split_axes[curr_id]] <= thresholds[curr_id]
        right = ~left

        left = indices[left]
        right = indices[right]

        stack.append((lefts[curr_id], left))
        stack.append((rights[curr_id], right))

    return res

class KDTree:

    def __init__(
        self,
        spatial_features: dict[str, list[float]],
        features_types: dict[str, str],
        handle_categorical=False,
    ):

        # We just remove categorical features from the bounds if the user wants to
        if not handle_categorical:
            spatial_features = {
                k: v
                for k, v in spatial_features.items()
                if features_types[k] != "Categorical"
            }

        self.ordering = sorted([k for k in spatial_features.keys()])
        self.nodes = [KDTreeNode(spatial_features, 0, tree=self)]
        self.index_map = {k: i for i, k in enumerate(self.ordering)}
        self.split_axis = np.array([-1], dtype=np.int32)
        self.thresholds =  np.array([-1], dtype=np.float32)
        self.lefts =  np.array([-1], dtype=int)
        self.rights =  np.array([-1], dtype=int)

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        assert all(
            k in X.columns for k in self.ordering
        ), f"X must contain the following columns: {self.ordering}"

        # Ensure the data is correctly ordered
        # And convert to numpy array for numba
        X = X[self.ordering].values
        X = np.asarray(X, dtype=np.float64)

        return _tree_predict(X, self.lefts, self.rights, self.split_axis, self.thresholds)
    
    def split(self, node: int, threshold, axis):

        bounds = self.nodes[node].bounds

        if bounds[axis][0] >= threshold or bounds[axis][1] <= threshold:
            raise ValueError("Threshold is outside the bounds of the node")
        
        left_bounds = bounds.copy()
        right_bounds = bounds.copy()

        right_bounds[axis] = [threshold, bounds[axis][1]]
        left_bounds[axis] = [bounds[axis][0], threshold]


        lid = len(self.nodes)
        lnode = KDTreeNode(left_bounds, id=lid, parent=node, tree=self)
        self.lefts[node] = lid

        rid = lid + 1
        rnode = KDTreeNode(left_bounds, id=rid, parent=node, tree=self)
        self.rights[node] = rid

        self.nodes.extend((lnode, rnode))
        self.lefts = np.append(self.lefts, (-1, -1))
        self.rights = np.append(self.rights, (-1, -1))

        self.thresholds[node] = threshold
        self.split_axis[node] = self.index_map[axis]

        self.thresholds = np.append(self.thresholds, [-1, -1])
        self.split_axis = np.append(self.split_axis, [-1, -1])

        return (lnode, rnode)