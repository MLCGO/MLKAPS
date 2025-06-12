"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.dummy
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mlkaps.configuration import ExperimentConfig


def plot_all_decision_tree(configuration: ExperimentConfig, decision_trees: dict):
    """
    Plot all the decision trees in the given dictionary, so the decision rules
    can be visualized.

    Parameters
    ----------
    configuration : ExperimentConfig
        The configuration of the experiment to plot the decision trees for.

    decision_trees : dict
        A dictionary of decision trees, with the key being the feature name
    """
    for feature, tree in decision_trees.items():
        plt.figure(figsize=(25, 20))
        plt.title("Decision Tree for " + feature)

        # If we're plotting classification trees, we need to specify the class names
        if isinstance(tree, sklearn.dummy.DummyClassifier):
            print("Skipping dummy classifier for feature: " + feature)
            continue
        elif isinstance(tree, DecisionTreeClassifier):
            class_names = [str(cl) for cl in tree.classes_]
            plot_tree(
                tree,
                impurity=False,
                proportion=True,
                feature_names=configuration.input_parameters,
                class_names=class_names,
                precision=2,
                filled=True,
            )
        else:
            # Else, just plot with the default settings
            plot_tree(tree, feature_names=configuration.input_parameters)

        output_path = configuration.output_directory / f"{feature}_decision_tree.png"
        plt.savefig(output_path)
        plt.close()


def plot_all_decisions_maps(
    configuration: ExperimentConfig,
    optimization_results: pd.DataFrame,
    clustering_model: dict,
):
    """
    Plot all the decisions maps for the given decisions models, as a heatmap/colormap with a
    scatter of the original optimization results.
    """
    if len(configuration.input_parameters) > 2:
        print("Clustering plotting is only available for 1D/2D input space")
        return

    for design_feature in configuration.design_parameters:
        plot_decision_map(clustering_model, configuration, optimization_results, design_feature)


def _create_colour_map(configuration, design_param):
    """
    Generate a colour mapping for a given feature, which can be used for both the decision
    heatmap/colormap and scatter plot
    """

    feature_type = configuration.parameters_type[design_param]
    feature_values = configuration["parameters"]["features_values"][design_param]
    value_count = len(feature_values)

    cmap = matplotlib.colormaps.get_cmap("YlGn")
    if feature_type in ["Categorical", "Boolean"]:
        from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        cmap = from_list(None, plt.cm.Set1(range(0, value_count)), value_count)

    return cmap


def plot_decision_map(clustering_model, configuration, optimization_results, design_param):
    """
    Plot a single decision map as a heatmap/colormap with a scatter of the original optimization
    results.
    """
    plt.figure(figsize=(15, 11), dpi=100)
    fig, ax = plt.subplots()

    # Set the plot title depending on the clustering method and parameters, and the design
    # parameter we're plotting
    _set_plot_title(ax, configuration, design_param)

    # Create a map that binds every possible value for this feature to a color
    color_map = _create_colour_map(configuration, design_param)

    # Plot the background colormap
    _plot_decision_colormap(clustering_model, configuration, design_param, optimization_results, color_map)

    # Plot the original optimization results as a scatter plot
    _scatter_optimization_results(ax, color_map, configuration, design_param, optimization_results)

    # Ensure the axis covers the entire range of the input features
    _set_axis_range(ax, configuration)

    _save_plot(configuration, design_param)
    plt.close()


def _set_plot_title(ax, configuration, design_param):
    """
    Set the title of the plot depending on the clustering method and parameters, and the design
    parameter we're plotting
    """

    clustering_parameters = configuration["clustering"]["clustering_parameters"]
    clustering_method = configuration["clustering"]["clustering_method"]
    param_title = ""
    for k, v in clustering_parameters.items():
        param_title += k + " = " + str(v) + " "
    title = f'Clustering model for parameter "{design_param}"\n' f"({clustering_method} - {param_title})"
    ax.set_title(title)


def _save_plot(configuration, design_param):
    """
    Save the plot to the output directoryP
    """

    # Ensure the output directory (and parents) exists
    os.makedirs(configuration.output_directory / "clustering/", exist_ok=True)
    output_path = configuration.output_directory / f"clustering/clustering_" f"{design_param}.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")


def _set_axis_range(ax, configuration):
    features_values = configuration["parameters"]["features_values"]
    input_parameters = configuration.input_parameters

    ax.set_xlabel(input_parameters[0])
    ax.set_xlim(features_values[input_parameters[0]][0], features_values[input_parameters[0]][1])

    if len(input_parameters) == 2:
        ax.set_ylabel(input_parameters[1])
        ax.set_ylim(
            features_values[input_parameters[1]][0],
            features_values[input_parameters[1]][1],
        )
    elif len(input_parameters) == 1:
        ax.set_ylabel(configuration["parameters"]["design"][0])
        plt.legend()


def _scatter_optimization_results(ax, color_map, configuration, design_param, optimization_results):
    feature_values = configuration["parameters"]["features_values"][design_param]
    y = optimization_results[design_param]
    values_y = np.unique(y)

    # Plot the optimization results on the same plot
    feature_type = configuration.parameters_type[design_param]
    input_parameters = configuration.input_parameters
    if feature_type in ["Categorical", "Boolean"]:
        inverse_value_map = {v: k for k, v in configuration["parameters"]["feature_values"][design_param].items()}
        y = np.vectorize(inverse_value_map.get)(y)

        if len(input_parameters) == 2:
            sc = ax.scatter(
                optimization_results[input_parameters[0]],
                optimization_results[input_parameters[1]],
                c=y,
                cmap=color_map,
                edgecolors="black",
                linewidths=1,
            )

            test_values = [fv for fv in feature_values if fv in values_y]
            plt.legend(
                handles=sc.legend_elements()[0],
                labels=test_values,
                loc="upper right",
                title=design_param,
            )
        elif len(input_parameters) == 1:
            ax.scatter(
                optimization_results[input_parameters[0]],
                y,
                s=30,
                color="b",
                label="training predictions",
            )

    else:  # int or float
        ax.scatter(
            optimization_results[input_parameters[0]],
            optimization_results[input_parameters[1]],
            c=np.array(y),
            cmap=color_map,
            vmin=feature_values[0],
            vmax=feature_values[1],
            edgecolors="black",
            linewidths=1,
        )


def _format_clustering_predictions(configuration, design_param, predictions, x_shape):
    feature_type = configuration.parameters_type[design_param]

    if feature_type == "int":
        predictions = np.array([np.round(x) for x in predictions])
    elif feature_type in ["Categorical", "Boolean"]:
        # We need to map categorial values to their index
        inverse_value_map = {v: k for k, v in configuration["parameters"]["feature_values"][design_param].items()}

        predictions = np.vectorize(inverse_value_map.get)(predictions)

    # pcolormesh requires a grid as an input, so we need to reshape the predictions
    # To fit the space defined by x_space/y_space
    return predictions.reshape(x_shape)


def _get_clustering_predictions(clustering_model, configuration, design_param, sampling_points):
    """
    Run the prediction model on the sampling points to get the clustering predictions
    We can then use those predictions to build a colormap
    """
    input_parameters = configuration.input_parameters

    x_min = sampling_points[input_parameters[0]].min() - 1
    x_max = sampling_points[input_parameters[0]].max() + 1
    x_npoints = min(100, int(np.ceil(x_max)))

    model_input = []
    samples = []
    if len(input_parameters) == 1:
        x_space = np.linspace(x_min, x_max, x_npoints)
        model_input = np.c_[x_space.ravel()]
        samples = [x_space]

    elif len(input_parameters) == 2:

        y_min = sampling_points[input_parameters[1]].min() - 1
        y_max = sampling_points[input_parameters[1]].max() + 1

        y_npoints = min(100, int(np.ceil(y_max)))
        x_space, y_space = np.meshgrid(np.linspace(x_min, x_max, x_npoints), np.linspace(y_min, y_max, y_npoints))
        # meshgrid(...) outputs a 2D array, while the clustering model expects
        # a 1d array
        model_input = np.c_[x_space.ravel(), y_space.ravel()]
        samples = [x_space, y_space]
    else:
        raise ValueError(f"Clustering error, expected 1D/2D input space, got {len(input_parameters)}D")
    model = clustering_model[design_param]
    predictions = model.predict(model_input)
    return predictions, samples


def _get_formatted_clustering_predictions(configuration, clustering_model, design_param, sampling_points):
    """
    Build a prediction grid that can be used for plotting the colormap
    First find the predictions,
    then format the samples as a grid that can be used for plotting
    """
    predictions, samples = _get_clustering_predictions(clustering_model, configuration, design_param, sampling_points)

    predictions = _format_clustering_predictions(configuration, design_param, predictions, samples[0].shape)

    return samples, predictions


def _plot_decision_colormap(clustering_model, configuration, design_param, optimization_results, color_map):
    """
    Plot a colormap using colormesh
    """
    input_parameters = configuration.input_parameters

    sampling_points = optimization_results[input_parameters]
    samples, predictions = _get_formatted_clustering_predictions(
        configuration, clustering_model, design_param, sampling_points
    )

    if len(input_parameters) == 2:
        feature_type = configuration.parameters_type[design_param]

        if feature_type in ["Categorical", "Boolean"]:
            plt.pcolormesh(samples[0], samples[1], predictions, cmap=color_map, alpha=1)
        else:
            feature_values = configuration["parameters"]["features_values"][design_param]
            plt.pcolormesh(
                samples[0],
                samples[1],
                predictions,
                cmap=color_map,
                alpha=1,
                vmin=feature_values[0],
                vmax=feature_values[1],
            )
            cbar = plt.colorbar()
            # cbar.ax.set_title(design_param)
            cbar.ax.set_title("parameter_1")

    elif len(input_parameters) == 1:
        plt.scatter(samples[0], predictions, color="b", alpha=0.4, label="testing predictions")
