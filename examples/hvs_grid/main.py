#!/usr/bin/env python3

from mlkaps.sampling.adaptive import HVSampler
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import matplotlib.patches as patch
import pandas as pd
from lightgbm import LGBMRegressor

kOutputPath = pathlib.Path("test_hvs_plots")


def makedirs():
    os.makedirs(kOutputPath, exist_ok=True)


def objective(input: pd.DataFrame) -> pd.DataFrame:
    func = lambda x: np.copysign(np.sin(x["x"]) ** 2 * np.cos(x["y"]) ** 2, x["x"])
    input["r"] = input.apply(func, axis=1)
    return input


def generate_grid(size):
    lx = [-10, 10]
    ly = [-10, 10]

    sx = np.linspace(lx[0], lx[1], size)
    sy = np.linspace(ly[0], ly[1], size)

    return pd.DataFrame(
        np.array(np.meshgrid(sx, sy)).T.reshape(-1, 2), columns=["x", "y"]
    )


def generate_baseline():
    df = objective(generate_grid(300))

    plt.figure(figsize=(8, 8), layout="constrained")
    plt.tripcolor(df["x"], df["y"], df["r"], cmap="YlGn")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True function")
    plt.savefig("true_function.png")
    plt.close()

    df = objective(generate_grid(50))

    return df


def plot_samples(plot_data, samples, partitions, tree, len_old_samples):

    old_samples = samples.iloc[:len_old_samples]
    new_samples = samples.tail(len(samples) - len_old_samples)

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), layout="constrained")
    ax = axs[0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Samples")
    ax.tripcolor(plot_data["x"], plot_data["y"], plot_data["r"], cmap="YlGn")

    for p in partitions:
        axes = p.axes

        width = axes["x"][1] - axes["x"][0]
        height = axes["y"][1] - axes["y"][0]

        rect = patch.Rectangle(
            [axes["x"][0], axes["y"][0]],
            width,
            height,
            edgecolor="black",
            alpha=0.4,
            fill=False,
        )
        ax.add_patch(rect)

    ax.scatter(old_samples["x"], old_samples["y"], c="black", marker="x", s=12)
    ax.scatter(new_samples["x"], new_samples["y"], c="red", marker="x", s=120)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax = axs[1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("HVS Decision Tree")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    df = generate_grid(100)
    df["r"] = tree.predict(df)
    ax.tripcolor(df["x"], df["y"], df["r"], cmap="YlGn")

    ax = axs[2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("LightGBM model")

    params = {
        "objective": "mae",
        "n_jobs": -1,
        "verbose": -1,
        "n_estimators": 400,
        "min_data_in_leaf": 20,
        "boosting": "gbdt",
        "learning_rate": 0.05,
    }
    model = LGBMRegressor(**params)
    model.fit(samples[["x", "y"]], samples["r"])

    df["r"] = model.predict(df[["x", "y"]])
    ax.tripcolor(df["x"], df["y"], df["r"], cmap="YlGn")

    fig.suptitle(f"Samples: {len(samples)}")
    path = kOutputPath / f"hvs_at_{len(samples)}.png"
    plt.savefig(path)
    plt.close()


def main():

    makedirs()

    inputs = {"x": [-10, 10], "y": [-10, 10]}
    types = {"x": "float", "y": "float"}

    sampler = HVSampler(types, inputs, error_metric="variance")
    samples = None

    plot_data = generate_baseline()

    while samples is None or len(samples) < 4000:
        len_before = 0 if samples is None else len(samples)

        if samples is None:
            # Take more samples during bootstrap
            n_samples = 400
        else:
            n_samples = 10

        print(f"At {len(samples) if samples is not None else 'bootstrap'} samples")

        samples = sampler.sample(n_samples, samples, objective)

        # If this is the first iteration, final partitions may not be set
        if sampler.final_partitions is None:
            continue

        plot_samples(
            plot_data, samples, sampler.final_partitions, sampler.final_tree, len_before
        )


if __name__ == "__main__":
    main()
