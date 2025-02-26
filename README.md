# MLKAPS 

MLKAPS (Machine Learning for Kernel Accuracy and Performance Studies) is a black-box machine learning auto-tuning tool predicting best design parameters of a computational kernel for any given input configuration.

It was succesfully used to tune software arithmetic, numerical kernels, as well as hardware arithmetic unit design, and competitive analysis.

To contribute, please open a new Github PR or get in contact with the contributors.
  
## Index

 * [Description](#description)
 * [Installation](#installation)
 * [Usage](#usage)
    * [Configuring an experiement](#creating-a-configuration-file)
    * [Launching an experiment](#launching-an-experiment)  
 * [Licence and copyright](#licence-and-copyright)  
 * [Cite this work](#cite-this-work)
 
## Description

Configuring applications to achieve maximum performance, accuracy, or  minimum energy consumption is a difficult task. It is often achieved via hand-tuning, which is subject to bias and blind-spots. MLKAPS attempts to solve this issue using auto-tuning techniques that leverages Machine Learning (ML).

MLKAPS is conceived to generate decision trees that give the best configuration to an input problem.


![MLKAPS workflow chart](doc/workflow.png)

This workflow represents MLKAPS pipeline. Given a kernel taking a set of input parameters I (matrix size, length of input vector, ...) and a set of design parameters D (number of threads, number of nodes, block size, tiling...), and an objective function Ob(i),  MLKAPS goal is to find a function F such that F(i)=d_min with d_min=argmin(Ob(I)), i.e. for each input in I return the design parameters in D that minimize the objective(s) Ob. Current releas focus on F being a decision tree that provide the best configuration trade-off for any input to easily translate into code to be used in high-performance libraries. 

The main steps of the pipeline for software kernel optimization are:
1. **Input configuration**: The users provides a JSON configuration files describing the application to tune, including the input and design parameters, and MLKAPS workflow configuration.
2. **Sampling**: MLKAPS samples the input and design parameters space using a variety of sampling algorithm, including adaptive sampling.
4. **Surrogate modeling**: MLKAPS builds regression models capable of cost-effectively predict the different objectives described in the configuration.
5. **Optimizing**: MLKAPS select pointss on the input space, and find the optimal configuration for each point.
6. **Building decision tree**: MLKAPS generate decision trees of given depth that are fitted on the optimization grid.
7. **(Experimental) Generating code**: the decision trees are exported to C code to be used in user application.

MLKAPS is quite versatil, and the various step can be repurposed for similar  optimization problems. On-going and future work will improve the interface and provide more pre-defined features and workflow.

## Installation

MLKAPS can be installed using pip. We recommend using virtualenv.

You can simply run the following command at the root of the sources:
```bash
pip install .
```

`mlkaps` should now be installed and added to your `PATH`. You can test the install was succesfull by running the tests:

```bash
pytest ./tests
```

or running

```
mlkaps --help
```


## Usage

In order to run, MLKAPS needs two inputs:

1. A benchmark/application executable. The input parameters will be given by MLKAPS sampler as argument to the script in the order defined in the configuration file. The output objectives (execution time, energy consumption, numerical accuracy) should be printed as a comma-separated list, in the last line of the executable output. We recommend wrapping the target in bash script if needed. **Ensure that this executable exits with -1 on failure.**
2. A configuration describing the experiments


### Creating a configuration file

For configuration description see [JSON configuration documentation](doc/json_conf.md). Ensure the parameters range and ordering is coherent between the configuration file and your executable.

Furture work is considering python interface to handle experiement definition, escaping the error prone process of carefuly writing list in json using matching ordering. (Issue #13)

### Launching an experiment

When your executable and configuration file are ready, you can run MLKAPS by using:
```bash
mlkaps -o <output_directory> <config.json> 
```

**We highly recommend using `tmux` or `screen` to prevent your experiment stopping if you lose connection to your tuning environment**. You should also validate your setup by running a simplified version of your kernel/configuration.

## Licence and copyright

* Copyright (C) 2020-2024 Intel Corporation
* Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
* Copyright (C) 2024-  MLKAPS contributors
* SPDX-License-Identifier: BSD-3-Clause

## Cite this work

[MLKAPS: Machine Learning and Adaptive Sampling for HPC Kernel Auto-tuning](https://arxiv.org/abs/2501.05811)

Mathys Jam (LI-PaRAD, UVSQ), Eric Petit (Intel), Pablo de Oliveira Castro (LI-PaRAD, UVSQ), David Defour (LAMPS, UPVD), Greg Henry, William Jalby (LI-PaRAD, UVSQ)


