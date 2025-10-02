Main page
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   user-guides/quick-start
   user-guides/pipeline

What is MLKAPS
---------------
Machine Learning for Kernel Accuracy and Performance Studies (MLKAPS) \
is a tool designed for automated kernel design parameters auto-tuning.

Ususaly, kernel optimization is a difficult task that requires a deep understanding of the problem at hand,
expert knowledge about the hardware, compilation tools, and the kernel itself. Optimizing a single kernel can take
months of work for a single engineer, and maintaining a large collection of kernels is a daunting task.

With the rapid growth of accelerators and new hardware architectures, long term support for math libraries
is becoming increasingly costly.

MLKAPS aims at providing an efficient way to explore and optimize kernel design parameter configuration for any set of input parameters, while requiring the least number of sample measurement.
After the initial setup, MLKAPS configuration files and scripts can be used to automatically maintain a kernel
over long period of time.

NOTE/FAQ
-------------------

**MLKAPS is not a code generator.**
It is a tool that help you find the best combination of design parameters
- so called optimization knobs - for any given input.

It will require you to write parametrized kernel and wrapper to explore input and design paramters configuration as described in the  configuration file for your experiement.

Available features
-------------------

While the tool is continuously extending its functionality, it already boast a number of feature, including:

* Various type of kernel parameters, including float, integers, boolean and categorial parameters
* Support for multi-objective optimization
* Basic support for compiler flags exploration
* Bootstrap sampling: Grid, random and latin hypercube kernel
* Adaptive sampling algorithm: HVS, HVSr, multilevel HVS and GA-adaptive
* Plotting scripts samples to visualize the objective model, the optimization space and the decision trees

Planned features
-------------------

* Adaptive sampling
* Improved results validation pipeline
* Improved integration with compiler toolchains

Copyrights
-------------------

* Copyright (C) 2020-2024 Intel Corporation
* Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
* Copyright (C) 2024-  MLKAPS contributors
* SPDX-License-Identifier: BSD-3-Clause
