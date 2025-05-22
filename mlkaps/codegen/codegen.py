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


class CGenerator:
    """Simple helper class to generate code in C."""

    def getType(self, x, types):
        return "const char *" if types[x] == "Categorical" else types[x]

    def genFunction(self, off, feature, inputs, types):
        return (
            f"{off}{self.getType(feature, types)}"
            f"get_{feature}({', '.join([f'{self.getType(f, types)} {f}' for f in inputs])}) {{\n"
        )

    def genIf(self, off, condition):
        return f"{off}if ({condition}) {{\n"

    def genElse(self, off):
        return f"{off}}} else {{\n"

    def genEnd(self, off):
        return f"{off}}}\n"

    def genReturn(self, off, lhs):
        return f"{off}return {lhs};\n"


class PythonGenerator:
    """Simple helper class to generate code in Python."""

    def getType(self, x, types):
        return "str" if types[x] == "Categorical" else types[x]

    def genFunction(self, off, feature, inputs, types):
        return f"{off}def get_{feature}({', '.join([f'{f}: {self.getType(f, types)}' for f in inputs])}):\n"

    def genIf(self, off, condition):
        return f"{off}if {condition}:\n"

    def genElse(self, off):
        return f"{off}else:\n"

    def genEnd(self, off):
        return ""

    def genReturn(self, off, lhs):
        return f"{off}return {lhs}\n"


def write_decision_trees(configuration: ExperimentConfig, decision_trees: dict):
    """
    Write the decision trees to a file in the specified language (C or Python).
    Generate one function per decision tree, each accepting kernel inputs as arguments.
    """
    if not configuration.tree_language:
        print("No tree language specified in the configuration. No trees will be generated.")
        return

    if configuration.tree_language == "Python":
        generator = PythonGenerator()
        output_path = configuration.output_directory / "decision_tree.py"
    else:
        generator = CGenerator()
        output_path = configuration.output_directory / "decision_tree.c"

    with open(output_path, "w") as file:
        for tree_key in decision_trees:
            write_decision_tree(configuration, decision_trees[tree_key], tree_key, generator, file)

    print(f"Decision Trees were written to {output_path}.")


def write_decision_tree(configuration: ExperimentConfig, decision_tree, feature, generator, file):
    """
    Write a single decision tree to a file in the specified language (C or Python).
    Generate a function for the decision tree, accepting kernel inputs as arguments.
    Use given code generator to generate the code.
    """
    feature_is_categorical = configuration.parameters_type[feature] == "Categorical"

    # Generate a function and recursively write the decision tree
    file.write(
        generator.genFunction("", feature, configuration.get_kernel_input_features(), configuration.get_all_features_types())
    )

    # If the tree is a dummy classifier, then we can't use the tree_ attribute
    # Generate a dummy prediction
    if isinstance(decision_tree, sklearn.dummy.DummyClassifier):
        prediction = decision_tree.predict([1])
        prediction = f'"{prediction[0]}"' if feature_is_categorical else prediction[0]
        file.write(generator.genReturn(" " * 4, prediction))
        file.write(generator.genEnd("") + "\n")
        return

    left = decision_tree.tree_.children_left
    right = decision_tree.tree_.children_right
    splitting_threshold = decision_tree.tree_.threshold
    # We need an array of the features for each node.
    features = [configuration.input_parameters[i] if i != -2 else None for i in decision_tree.tree_.feature]
    value = decision_tree.tree_.value

    if isinstance(decision_tree, DecisionTreeClassifier):
        classes = decision_tree.classes_
    else:
        classes = None

    def recurse(generator, left, right, threshold, features, node, output, depth):
        offset = " " * 4 * depth

        # Check if the current node is not a leaf node
        if threshold[node] != -2:
            output.write(generator.genIf(offset, f"{features[node]} <= {threshold[node]}"))
            if left[node] != -1:
                recurse(generator, left, right, threshold, features, left[node], output, depth + 1)
            output.write(generator.genElse(offset))
            if right[node] != -1:
                recurse(generator, left, right, threshold, features, right[node], output, depth + 1)
            output.write(generator.genEnd(offset))
        else:
            # Handle leaf node
            feature_value = value[node]
            if classes is not None:
                feature_value = f"{classes[np.argmax(value[node])]}"
            else:
                feature_value = feature_value[0][0]
            if feature_is_categorical:
                feature_value = f'"{feature_value}"'
            output.write(generator.genReturn(offset, feature_value))

        return output

    recurse(generator, left, right, splitting_threshold, features, 0, file, 1)
    file.write(generator.genEnd("") + "\n")
