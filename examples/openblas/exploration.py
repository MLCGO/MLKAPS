#!/usr/bin/env python3
import numpy as np
import pandas as pd
import psutil

from mlkaps.sample_collection import MonoSubprocessHarness, MonoKernelExecutor, FailedRunResolver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Juste redefine the variables used in the experiment
ncores = len(psutil.Process().cpu_affinity())
print(psutil.Process().cpu_affinity())
feature_values = {
    "vecsize": [256, 20000],
    "nthreads": [1, ncores]
}
input_features = ["vecsize"]
design_parameters = ["nthreads"]
# This should match the order expected by the kernel
kernel_parameters = ["vecsize", "nthreads"]

# == Build an MLKAPS runner so we can evaluate new samples ==
#harness = MonoSubprocessHarness(["performance"], 
#                                "./openblas_kernel/build/openblas_kernel",
#                                #"./dummy.sh",
#                                kernel_parameters,
#                                timeout=30)

# Define the objectives bounds
objectives_bounds = {"performance": 3}

# == Build an MLKAPS runner so we can evaluate new samples ==
harness = MonoSubprocessHarness(
    objectives=["performance"], 
    objectives_bounds= objectives_bounds,
    executable_path="./openblas_kernel/build/openblas_kernel",
    arguments_order=kernel_parameters,
    timeout=30
)

resolver = FailedRunResolver.from_name("discard")
runner = MonoKernelExecutor(harness, resolver, progress_bar=True)

class TreeWrapper:

    def __init__(self):
        self.trees = None
        self.ordering = None

    def fit(self, X, y):
        self.ordering = sorted(X.columns)
        X = X[self.ordering]
        self.trees = {}

        from sklearn.tree import DecisionTreeRegressor
        for col in y.columns:
            tree = DecisionTreeRegressor()
            tree.fit(X, y[col])
            self.trees[col] = tree

    def predict(self, X):
        X = X[self.ordering]
        res = pd.DataFrame({col: tree.predict(X) for col, tree in self.trees.items()})
        return pd.concat([X, res], axis=1)

def reload_trees():
    res = {}
    path = Path("runs")

    for file in path.iterdir():
        if not file.is_dir():
            continue
        if not (file / "optim.csv").exists():
            continue

        run_name = file.name.lower().replace("_"," ")
        run = pd.read_csv(file / "optim.csv")
        tree = TreeWrapper()
        tree.fit(run[input_features], run[design_parameters])
        res[run_name] = tree

    return res

def do_experiment(exploration_path, runs_path):
    vecsizes = np.linspace(256, 20000, 20).astype(int)
    ranges = np.arange(1, ncores + 1)

    df = pd.DataFrame(np.array(np.meshgrid(vecsizes, ranges)).T.reshape(-1, 2), columns=kernel_parameters)
    samples = runner(df)

    # next, Fetch mlkaps trees

    to_test = pd.DataFrame({"vecsize": vecsizes})
    tree = reload_trees()
    r = None
    for run_name, tree in tree.items():
        csamples = tree.predict(to_test)
        csamples = runner(csamples)
        csamples["run"] = run_name

        if r is None:
            r = csamples
        else:
            r = pd.concat([r, csamples])

    r.to_csv(runs_path, index=False)
    samples.to_csv(exploration_path, index=False)

    return r, samples




def main():
    if len(sys.argv) != 2:
        print("Usage: python exploration.py <output>")
        sys.exit(1)
    
    output_path = Path(sys.argv[1])
    output_path.mkdir(parents=True, exist_ok=True)

    exploration_path = output_path / "exploration.csv"
    runs_path = output_path / "runs_samples.csv"

    if not exploration_path.exists() or not runs_path.exists():
        runs, samples = do_experiment(exploration_path, runs_path)
    else:
        runs = pd.read_csv(runs_path)
        samples = pd.read_csv(exploration_path)

    with plt.rc_context(rc={"font.size": 10}):
        # We want one lineplot per vector size
        fig, axs = plt.subplots(4, 4, figsize=(16, 16), layout="constrained")
        colormap = matplotlib.colormaps.get_cmap('tab10')  # Get the colormap
        colors = [colormap(i) for i in range(len(runs["run"].unique()))]  # Generate colors
        for ax, size in zip(axs.flatten(), np.unique(samples["vecsize"])):
            subset = samples[samples["vecsize"] == size]
            ax.plot(subset["nthreads"], subset["performance"], marker="o")
            ax.set_title(f"Vector size: {size}")
            ax.set_xlabel("Number of threads")
            ax.set_ylabel("Execution Time (s)")

            for i, run_name in enumerate(runs["run"].unique()):
                run = runs[runs["run"] == run_name]
                run = run[run["vecsize"] == size]
                ax.axvline(run["nthreads"].iloc[0], color=colors[i], linestyle="--", label=run_name)
            ax.legend(loc="upper right")

        fig.savefig(output_path / "exploration.png")


if __name__ == "__main__":
    main()