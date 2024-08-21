#!/usr/bin/env python3

from mlkaps.sampling import HVSampler
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import matplotlib.patches as patch
import pandas as pd
from lightgbm import LGBMRegressor
import multiprocessing
from tqdm import tqdm
import argparse

def objective(input: pd.DataFrame) -> pd.DataFrame:
    func = lambda x: np.sin(abs(x["x"] * x["y"]))
    input["r"] = input.apply(func, axis=1)
    return input


def generate_grid(size):
    lx = [-2, 2]
    ly = [-10, 2]

    sx = np.linspace(lx[0], lx[1], size)
    sy = np.linspace(ly[0], ly[1], size)

    return pd.DataFrame(
        np.array(np.meshgrid(sx, sy)).T.reshape(-1, 2), columns=["x", "y"]
    )


def generate_baseline():
    df = objective(generate_grid(300))

    plt.figure(figsize=(8, 8), layout="constrained")
    plt.tripcolor(df["x"], df["y"], df["r"], cmap="YlGn")
    plt.xlim(-2, 2)
    plt.ylim(-10, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True function")
    plt.savefig("true_function.png")
    plt.close()

    df = objective(generate_grid(50))

    return df


def plot_samples(args, plot_data, samples, partitions, tree, last=False):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")

    ax = axs[0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("HVS Decision Tree")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-10, 2)

    df = generate_grid(100)
    df["r"] = tree.predict(df[["x", "y"]])

    ax.tripcolor(df["x"], df["y"], df["r"], cmap="binary")

    ax = axs[1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Samples")
    ax.tripcolor(plot_data["x"], plot_data["y"], plot_data["r"], cmap="binary")

    if not last:
        for p in partitions:
            axes = p.axes
            width = axes["x"][1] - axes["x"][0]
            height = axes["y"][1] - axes["y"][0]

            rect = patch.Rectangle(
                [axes["x"][0], axes["y"][0]],
                width,
                height,
                edgecolor="red",
                alpha=0.4,
                fill=False,
            )
            ax.add_patch(rect)

    ax.scatter(samples["x"], 
               samples["y"], c=samples["r"], marker=".", 
               s=130, edgecolor="none", cmap="magma")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-10, 2)

    ax = axs[2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-10, 2)
    ax.set_title("LightGBM model")
    params = {
        "objective": "mae",
        "n_jobs": 1,
        "verbose": -1,
        "n_estimators": 400,
        "min_data_in_leaf": 20,
        "boosting": "gbdt",
        "learning_rate": 0.05,
    }
    model = LGBMRegressor(**params)
    model.fit(samples[["x", "y"]], samples["r"])

    df["r"] = model.predict(df[["x", "y"]])
    ax.tripcolor(df["x"], df["y"], df["r"], cmap="binary")

    fig.suptitle(f"Samples: {len(samples)}")

    n_digits = int(np.log10(args.nsamples)) + 1
    tmp = str(len(samples)).zfill(n_digits)
    path = args.output / f"hvs_at_{tmp}{'_last' if last else ''}.png"
    plt.savefig(path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="2D Example of HVS Sampling")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output path", default="output")
    parser.add_argument("-n", "--nsamples", type=int, help="Number of samples to take", default=6000)
    parser.add_argument("--nbootstrap", type=int, help="Number of samples to take during bootstrap", default=400)
    parser.add_argument("--per-iter", type=int, help="Number of samples to take during bootstrap", default=10)

    parser.add_argument("--nworkers", type=int, help="Number of workers to use", default=multiprocessing.cpu_count())

    args = parser.parse_args()

    # Ensure the output path exists
    args.output.mkdir(parents=True, exist_ok=True)

    return args


def main():

    args = parse_args()

    inputs = {"x": [-2, 2], "y": [-10, 2]}
    types = {"x": "float", "y": "float"}

    sampler = HVSampler(types, inputs, error_metric="variance")
    samples = None

    plot_data = generate_baseline()

    with multiprocessing.Pool(args.nworkers) as pool, tqdm(total=args.nsamples, desc="Sampling", leave=None) as pbar:
        tasks = []
        while samples is None or len(samples) < args.nsamples:
            n_samples = args.nbootstrap if samples is None else args.per_iter

            samples = sampler.sample(n_samples, samples, objective)

            pbar.update(n_samples)

            # If this is the first iteration, final partitions may not be set
            if sampler.final_partitions is None:
                continue

            new_task = pool.apply_async(
                plot_samples,
                (args, plot_data, samples, sampler.final_partitions, sampler.final_tree),
            )
            tasks.append(new_task)

        pbar.set_description("Sampling done, awaiting plots")

        for t in tqdm(tasks, desc="Plotting...", leave=None):
            t.wait()

        pool.apply_async(
            plot_samples,
            (args, plot_data, samples, sampler.final_partitions, sampler.final_tree),     
            {"last": True},
        )
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
