# Introduction

This tutorial showcases how a very simple MLKAPS experiment is setup, and will guide you through all the steps to make your own.
The goal is to tune the number of threads for a simple outer product using OpenBLAS OpenMP.

The parameters are the following:

**Inputs**:
- `vecsize` $\in [256, 20000]$ the size of the vectors for the outerproduct

**Design parameters**:
- `nthreads` $\in [1, \text{max\_threads}]$ the number of threads used by OpenBLAS. 



## Pre-requisites

### Dependencies

You must have a C++ compiler with OpenMP support, and an OpenMP-enabled version of OpenBLAS. We use cmake to search for `libopenblaso.so` to locate the library.

**Fedora**

```sh
sudo dnf install openblas-devel
```

### Building the kernel

To validate your setup, compile the kernel by using:
```sh
cd ./openblas_kernel
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

And executing the kernel on a small test:
```sh
./openblas_kernel 256 1
# <output_time>
```

### Setting up the Python environment

Ensure you have a recent version of python 3 installed, along with pip and `virtualenv`

**Fedora**
```sh
sudo dnf install python3 python3-pip python3-virtualenv
```

Then, install all the required packages along with MLKAPS using:
```sh
./setup_all.sh
```

### Stabilizing your environment

The results of this experiment will depend on your hardware and software environment. Please ensure you have a clean and stable machine for the run.

Try running the following case a few time and ensure you have stable timings:
```sh
./openblas_kernel 8000 1
```

Try running the following cases, and check the scalability:
```sh
./openblas_kernel 8000 1
./openblas_kernel 8000 2
./openblas_kernel 8000 4
./openblas_kernel 8000 8
```

If the experiment is unstable, you should tune OpenMP flags, threads affinity, and other environment parameters to reach stability. If you're using a laptop, ensure you are running in performance mode.

An an example, on the author's laptop, the experiment is run using:
```sh
OMP_PLACES="{0,2,4,6,8,10,12,14}" OMP_PROC_BIND="spread" taskset -c 0,2,4,6,8,10,12,14 \
./openblas_kernel 8000 8
```
This ensures the kernel runs on the P-cores and that OpenMP correctly dispatches the tasks amongst the cores.


## MLKAPS & The experiment

### Content
We provide the following files to run the experiment:

- `run_all.sh` This is the main entry point of the experiment. It runs MLKAPS and performs additional experiments to validate MLKAPS results.
- `config_mlkaps_template.json` MLKAPS configuration files, describing the experiment. Note that we use `@max_threads` as a placeholder for the maximum number of threads to explore. **You should not overwite this placeholder, it will be done automatically.**
- `setup_all.sh` This file ensures the kernel is compiled and installs all python dependencies.
- `exploration.py` Performs an exhaustive search on the design space and validates MLKAPS results
- `templatize.py` Refreshes the template configuration with the actual number of threads

### Running the experiments

Running the experiment is a simple as running
```sh
./run_all.sh <run_label>
```
**Note that you should customize this command to your need.
On the author's laptop, the experiment is instead run using:**
```sh
OMP_PLACES="{0,2,4,6,8,10,12,14}" OMP_PROC_BIND="spread" taskset -c 0,2,4,6,8,10,12,14 \
./run_all.sh my_first_run
```

With 8 cores available, the experiments runs in around 20 minutes. It may take longer depending on your hardware, and the number of cores you allow the experiment to use.


### Output

After running the experiment, you get a few output:
- `runs/` will contain the MLKAPS generated-files, including the samples, optimums found, the surrogate model, and the decision trees. The content of this folder may depend on your version of MLKAPS.
- `results/` will contain the output of `exploration.py`, the results of the exhaustive search, and a plot showing how MLKAPS performed on this experiment.