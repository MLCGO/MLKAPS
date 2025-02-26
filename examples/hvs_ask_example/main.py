#!python3

from numpy import sin, pi
import numpy as np
from mlkaps.sampling.adaptive import HVSampler
import matplotlib.pyplot as plt
import pandas as pd

"""
The following tests check that HVS behave the same as the original version
Those are not truly automated tests, as they do not verify the output, but can be used
to visually check that the output is the same as the original version

Since they call HVS on both 1D and 2D problem, we keep them here since they can still raise 
errors if HVS is broken

"""


def objective(df: pd.DataFrame):
    df["objective"] = df.apply(
        lambda x: x["a"] ** 5 * abs(sin(6 * pi * x["a"])), axis=1
    )
    return df


features = {"a": [0, 1]}
ftypes = {"a": "float"}

sampler = HVSampler(ftypes, features, min_samples_per_leaf=5)
data = None

fig, axs = plt.subplots(5, 2, figsize=(10, 20), layout="constrained")

true_func = objective(pd.DataFrame(np.linspace(0, 1, 10000), columns=["a"]))
for i in range(10):
    ax = axs[i // 2][i % 2]
    if i == 0:
        last = 5
    else:
        last = 10

    data = sampler.sample(last, data, objective)

    ax.plot(true_func["a"], true_func["objective"], zorder=-1)
    ax.scatter(
        data["a"].iloc[:-last],
        data["objective"].iloc[:-last],
        label="Samples",
        color="black",
        marker="x",
        alpha=1.0,
                zorder=1.0

    )

    ax.scatter(
        data["a"].iloc[-last:],
        data["objective"].iloc[-last:],
        label="New samples",
        color="red",
        marker="x",
        alpha=1.0,
        zorder=1.0
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(true_func["objective"].min() * 0.9, true_func["objective"].max() * 1.1)
    ax.set_title(f"{len(data)} samples", fontweight="bold")

    if sampler.final_partitions is None:
        ax.legend()
        continue

    # Plot the final tree constraints
    for i, p in enumerate(sampler.final_partitions):
        axes = p.axes
        rect = plt.Rectangle(
            (axes["a"][0], 0),
            axes["a"][1] - axes["a"][0],
            1,
            color="green",
            alpha=0.2,
            label="HVS Partition(s)" if i == 0 else None,
        )
        ax.add_patch(rect)

    ax.legend()
    

fig.suptitle("HVS sampler\n"
             r"$f(x) = x^5 |sin(6 \pi x)|$", fontweight="bold", fontsize=16)
plt.savefig("ask1D.png")
