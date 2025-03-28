# OpenBlas

This tutorial showcases how a very simple [MLKAPS](https://github.com/MLCGO/MLKAPS) experiment is setup, and will guide you through all the steps to make your own.
The goal is to tune the number of threads for a simple outer product `dger` using OpenBLAS OpenMP.

The parameters are the following:

**Inputs**:

- `vecsize` $\in [256, 20000]$ the size of the vectors for the outer-product

**Design parameters**:

- `nthreads` $\in [1, \text{max\_threads}]$ the number of threads used by OpenBLAS

## Pre-requisites

For this experiment, we assume you already have MLKAPS installed inside a [virtualenv](https://virtualenv.pypa.io/en/latest/).

### Dependencies

You must have a C++ compiler with OpenMP support, and an OpenMP-enabled version of [OpenBLAS](http://www.openmathlib.org/OpenBLAS/). We use [CMake](https://cmake.org/) to search for `libopenblaso.so` to locate the library.

**Fedora**

```sh
sudo dnf install openblas-devel
```

**Ubuntu**

```bash
sudo apt install libopenblas-dev
```

### Building the kernel

To validate your setup, compile the kernel by using:

```bash
cd ./openblas_kernel
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

And executing the kernel on a small test:

```bash
./openblas_kernel 256 1
# <output_time>
```

### Stabilizing your environment

The results of this experiment will depend on your hardware and software environment. Please ensure you have a clean and stable machine for the run.

Try running the following case a few time and ensure you have stable timings:

```bash
./openblas_kernel 8000 1
```

Try running the following cases, and check the scalability:

```bash
./openblas_kernel 8000 1
./openblas_kernel 8000 2
./openblas_kernel 8000 4
./openblas_kernel 8000 8
```

If the experiment is unstable, you should tune OpenMP flags, threads affinity, and other environment parameters to reach stability. If you're using a laptop, ensure you are running in performance mode.

An an example, on the author's laptop, the experiment is run using:

```bash
OMP_PLACES="{0,2,4,6,8,10,12,14}" OMP_PROC_BIND="spread" taskset -c 0,2,4,6,8,10,12,14 \
./openblas_kernel 8000 8
```

This ensures the kernel runs on the P-cores and that OpenMP correctly dispatches the tasks amongst the cores.

## MLKAPS & The experiment

### Content

We provide the following files to run the experiment:

- `run_all.sh` This is the main entry point of the experiment. It runs MLKAPS and performs additional experiments to validate MLKAPS results.
- `config_mlkaps_template.json` MLKAPS configuration file, describing the experiment. Note that we use `@max_threads` as a placeholder for the maximum number of threads to explore. **You should not overwite this placeholder, it will be done automatically.**
- `exploration.py` Performs an exhaustive search on the design space and validates MLKAPS results
- `templatize.py` Refreshes the template configuration with the actual number of threads

### Running the experiments

Running the experiment is a simple as running

```bash
./run_all.sh <run_label>
```

**Note that you should customize this command to your need.
On the author's laptop, the experiment is instead run using:**

```bash
OMP_PLACES="{0,2,4,6,8,10,12,14}" OMP_PROC_BIND="spread" taskset -c 0,2,4,6,8,10,12,14 \
./run_all.sh my_first_run
```

With 8 cores available, the experiments runs in around 20 minutes. It may take longer depending on your hardware, and the number of cores you allow the experiment to use.

### Output

After running the experiment, you get the following outputs:

- `runs/<run_label>` will contain MLKAPS generated-files.
    - `kernel_sampling/samples.csv` contain the samples collected by MLKAPS
    - `metadata/` will contain the logs and a copy of the configuration used for the run.
    - `optim.csv` will contain the results of MLKAPS optimization phase: those are the optimums solutions predicted on a regular grid on the Input space. The final decision trees are trained on this data.
    - `performance_model.pkl` contains the surrogate model used by MLKAPS for the `performance` objective. You can easily reload this model using 
    [Pickle](https://docs.python.org/3/library/pickle.html) and call the `model.predict(data: pd.Dataframe)` method to predict new points. This is useful to validate the model.
    - `nthreads_clustered_model.pkl` contains the decision tree fitted on `optim.csv` ([Scikit-Learn decision tree regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)).
    - `C_nthreads_tree.c`contains the decision tree converted to C code.

**Note: pickled objects may not be compatible between python versions, and can easily break when updating the virtualenv. It is common to refit the different models using the raw csv data.**

**`exploration.py` shows how to refit the decision trees from `optim.csv`.**

- `results/` will contain the output of `exploration.py`.
    - `exploration.csv` contains the exhaustive search on different vector size and number of threads. This can be useful to validate MLKAPS' choices.
    - `exploration.png` shows the results of the exhaustive search.
    - `runs_samples.csv` shows the results of running MLKAPS' decision tree for different size.
