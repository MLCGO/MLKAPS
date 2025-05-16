"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

"""
This file contains an implementation of a KDTree to be used with the bayesian sampler.
As opposed to classical usage of the KDTree, we do not use the KDTree to find nearest neighbors, 
but rather to partition the space into hyperrectangles.
"""

import numpy as np
import pandas as pd
from typing import Union
import logging

logger = logging.getLogger(__name__)

# If Numba is available, we use it to speed up the tree traversal
try:
    import numba

    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    import warnings

    warnings.warn("Numba is not available. JIT compilation will be disabled.", ImportWarning)


# Define a decorator to conditionally apply JIT compilation
def maybe_jit(*args, **kwargs):
    def decorator(func):
        return numba.jit(*args, **kwargs)(func) if JIT_AVAILABLE else func

    return decorator


class KDTreeNode:
    """
    A node in the KDTree. Each node is defined by its bounding hypercube and a unique ID.
    The node contains an opaque data field that can be used to store any additional information.

    In order to maximise performance, many fields are stored in flat arrays in the parent KDTree.
    """

    __slots__ = ["id", "parent", "tree", "data", "bounds", "depth"]

    id: int
    parent: None | int
    tree: "KDTree"
    data: None | object
    depth: int

    def __init__(
        self,
        bounds: dict[str, list[float]],
        id: int,
        parent: None | int,
        tree: Union["KDTree"],
    ):
        """Initialize a new KDTreeNode.

        :param bounds: A dictionnary containg the bounds of the node for each dimension.
        The keys are the names of the dimensions, and the values are lists of two floats for ub/lb.
        :type bounds: dict[str, list[float]]
        :param id: The UID of the node in the KDTree.
        :type id: int
        :param parent: The id of the parent node in the KDTree, or None if this is the root node.
        :type parent: None | int
        :param tree: A reference to the KDTree that contains this node.
        If None, the parent must be provided so the tree can be inferred.
        :type tree: KDTree | None
        """
        self.bounds = bounds

        self.id = id
        assert tree is not None, "Tree must be provided"

        self.parent = parent
        self.tree = tree

        self.data = None
        self.depth = 0 if parent is None else tree.nodes[parent].depth + 1

    @property
    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf node, False otherwise.

        :return: Returns True if the node is a leaf node, False otherwise.
        :rtype: bool
        """
        return self.tree.lefts[self.id] == -1 and self.tree.rights[self.id] == -1

    @property
    def volume(self) -> float:
        """Compute the volume of the hypercube defined by the node.

        :return: The volume of the bounding hypercube.
        :rtype: float
        """
        size = 1.0
        for k, (low, high) in self.bounds.items():
            if high == low:
                logger.warning(f"Warning: The bounds for the dimension {k} are equal. This will result in a volume of 0.")
            size *= high - low
        return size

    def split(self, axis: str, threshold: float) -> tuple["KDTreeNode", "KDTreeNode"]:
        """Split the node along the given axis at the given threshold.

        Redirects the call to the tree.

        :param axis: The name of the axis to split along.
        :type axis: str
        :param threshold: The value at which to split the node.
        Note that the threshold must be within the bounds of the node.
        :type threshold: float
        :return: The two new nodes created by the split.
        :rtype: tuple[KDTreeNode, KDTreeNode]
        :raises ValueError: If the threshold is outside the bounds of the node.
        """
        return self.tree.split(self.id, axis, threshold)


@maybe_jit(nogil=True)
def _tree_predict(
    X: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    split_axes: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Helper static function to predict the containing leaf for each point in X.
    Will use numba JIT if available.

    :param X: The points to be predicted.
    :type X: np.ndarray
    :param lefts: An array containing the left child of each node.
    :type lefts: np.ndarray
    :param rights: An array containing the right child of each node.
    :type rights: np.ndarray
    :param split_axes: An array containing the axis along which each node is split.
    The axis is given as an index in the original data.
    :type split_axes: np.ndarray
    :param thresholds: An array containing the threshold at which each node is split.
    :type thresholds: np.ndarray
    :return: An array containing the id of the leaf node containing each point in X.
    :rtype: np.ndarray
    """
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
    """An implementation of a KDTree to be used with the bayesian sampler.
    Most of the fields are stored in flat arrays to improve performance.

    This implementation is not a classical KDTree, as it does not use the tree to find nearest neighbors,
    but rather to partition the space into hyperrectangles, so that new points can be assigned to the correct partition.

    Each partition can contain a user-defined data field, which can be used to store any value.

    This KDTree is not balanced, and will use numba if available. The KDTree must be manually constructed by the user.

    >>> from mlkaps.sampling.bayesian.kdtree import KDTree
    >>> spatial_features = {
    ...     "x": [0, 1],
    ...     "y": [0, 1],
    ... }
    >>> features_types = {
    ...     "x": "float",
    ...     "y": "float",
    ... }
    >>> kdtree = KDTree(spatial_features, features_types)
    >>> childs = kdtree.split(0, "x", 0.5) # Split the root node (0) at x=0.5
    >>> kdtree.thresholds[0]
    np.float64(0.5)
    >>> kdtree.split_axis[0]
    np.int64(0)
    >>> kdtree.lefts[0]
    np.int64(1)
    >>> kdtree.rights[0]
    np.int64(2)
    """

    def __init__(
        self,
        spatial_features: dict[str, list[float]],
        features_types: dict[str, str],
        handle_categorical=False,
    ):
        """Initialize a new KDTree.

        :param spatial_features: A dictionnary containing the bounds of the node for each dimension.
        The keys are the names of the dimensions, and the values are lists of two floats for ub/lb.
        :type spatial_features: dict[str, list[float]]
        :param features_types: A dictionnary containing the type of each feature.
        :type features_types: dict[str, str]
        :param handle_categorical: Enables or disable the support of splitting upon categorical features, defaults to False
        :type handle_categorical: bool, optional
        """

        # We just remove categorical features from the bounds if the user wants to
        if not handle_categorical:
            spatial_features = {k: v for k, v in spatial_features.items() if features_types[k] != "Categorical"}

        # Ensure consistent ordering of the features
        self.ordering = sorted([k for k in spatial_features.keys()])
        # Map to convert from the feature name to the index in the original data
        self.index_map = {k: i for i, k in enumerate(self.ordering)}

        self.nodes = [KDTreeNode(spatial_features, 0, None, tree=self)]

        # Initialize the flat arrays to store the tree structure
        # -1 in the split axis, thresholds, lefts and rights arrays
        # indicates that the node is a leaf node and does not have the corresponding value
        self.split_axis = np.array([-1], dtype=np.int32)
        self.thresholds = np.array([-1], dtype=np.float32)
        self.lefts = np.array([-1], dtype=np.int32)
        self.rights = np.array([-1], dtype=np.int32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the id of the leaf node containing each point in X.

        :param X: A Pandas DataFrame containing the points to be predicted.
        All dimensions contained in self.ordering must be present in the DataFrame.
        :type X: pd.DataFrame
        :return: An array containing the id of the leaf node containing each point in X.
        :rtype: np.ndarray
        """

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        assert all(k in X.columns for k in self.ordering), f"X must contain the following columns: {self.ordering}"

        # Ensure the data is correctly ordered
        # And convert to numpy array for numba
        X = X[self.ordering].to_numpy(dtype=np.float32)

        return _tree_predict(X, self.lefts, self.rights, self.split_axis, self.thresholds)

    def split(self, node: int, axis: str, threshold: float) -> tuple[KDTreeNode, KDTreeNode]:
        """Split the provided node (id) at the given threshold along the given axis.

        :param node: The id of the node to split
        :type node: int
        :param axis: The name of the dimensions to split along
        :type axis: str
        :param threshold: The value at which to split the node.
        :type threshold: float
        :raises ValueError: raised if the threshold is outside the bounds of the node.
        :return: The newly created nodes
        :rtype: tuple[KDTreeNode, KDTreeNode]e
        """

        if self.lefts[node] != -1 or self.rights[node] != -1:
            raise ValueError(
                f"Node {node} is already split ({self.lefts[node]/self.rights[node]}, along {self.split_axis[node]}: {self.thresholds[node]})"
            )

        if node < 0 or node >= len(self.nodes):
            raise ValueError(f"Node {node} does not exist in the tree")

        if axis not in self.index_map:
            raise ValueError(f"Axis {axis} is not a valid axis. Valid axes are: {self.ordering}")

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
        rnode = KDTreeNode(right_bounds, id=rid, parent=node, tree=self)
        self.rights[node] = rid

        self.nodes.extend((lnode, rnode))
        self.lefts = np.append(self.lefts, (-1, -1))
        self.rights = np.append(self.rights, (-1, -1))

        self.thresholds[node] = threshold
        self.split_axis[node] = self.index_map[axis]

        self.thresholds = np.append(self.thresholds, [-1, -1])
        self.split_axis = np.append(self.split_axis, [-1, -1])

        return (lnode, rnode)
