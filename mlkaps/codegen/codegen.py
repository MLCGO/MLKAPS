"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.dummy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from mlkaps.configuration import ExperimentConfig


def plot_decision_tree(configuration: ExperimentConfig, decision_trees: dict):
    for tree_key in decision_trees:
        plt.figure(figsize=(25, 20))

        t = decision_trees[tree_key]
        if isinstance(t, DecisionTreeClassifier):
            clas_str = [str(cl) for cl in t.classes_]
            tree.plot_tree(
                t,
                impurity=False,
                proportion=True,
                feature_names=configuration.ginput_parameters,
                class_names=clas_str,
                precision=2,
                filled=True,
            )
        else:
            tree.plot_tree(t, feature_names=configuration.input_parameters)

        plt.savefig(str(tree_key) + "_decision_tree.png")
        plt.close()


def decision_trees_to_c(configuration: ExperimentConfig, decision_trees: dict):
    for tree_key in decision_trees:
        # If the tree is a dummy classifier, then we can't use the tree_ attribute
        # Generate a dummy prediction
        if isinstance(decision_trees[tree_key], sklearn.dummy.DummyClassifier):
            prediction = decision_trees[tree_key].predict([])
            print(f"{tree_key} = {prediction};\n")
        else:
            decision_tree_to_c(configuration, decision_trees[tree_key], tree_key)


def decision_tree_to_c(configuration: ExperimentConfig, decision_tree, feature):
    left = decision_tree.tree_.children_left
    right = decision_tree.tree_.children_right
    splitting_threshold = decision_tree.tree_.threshold
    # We need an array of the features for each node.
    features = [
        configuration.input_parameters[i] if i != -2 else None
        for i in decision_tree.tree_.feature
    ]
    value = decision_tree.tree_.value

    if isinstance(decision_tree, DecisionTreeClassifier):
        classes = decision_tree.classes_
    else:
        classes = None

    feature_is_categorical = configuration.parameters_type[feature] == "Categorical"
    output_str = ""

    def recurse(output_str, left, right, threshold, features, node, depth=0):
        offset = "\t" * depth

        # if the treshold is -2, then this is a leaf node
        if threshold[node] != -2:
            output_str += (
                offset
                + "if ("
                + features[node]
                + " <= "
                + str(threshold[node])
                + ") {\n"
            )
            if left[node] != -1:
                output_str = recurse(
                    output_str, left, right, threshold, features, left[node], depth + 1
                )
            output_str += offset + "} else {\n"

            if right[node] != -1:
                output_str = recurse(
                    output_str, left, right, threshold, features, right[node], depth + 1
                )
            output_str += offset + "}\n"
        else:
            feature_value = value[node]

            if classes is not None:
                feature_value = f"{classes[np.argmax(value[node])]}"
            else:
                feature_value = feature_value[0][0]
            if feature_is_categorical:
                feature_value = f'"{feature_value}"'

            output_str += f"{offset} {feature} = {feature_value};\n"
        return output_str

    output_str = recurse(output_str, left, right, splitting_threshold, features, 0, 0)
    print(f'Decision Tree for "{feature}":')
    print(output_str)
