Writing a configuration file
==================================

The kernel description need to be written in a JSON configuration file that describes our kernel to MLKAPS. The following sections describes the different sections of a typical configuration file.


--------------------------------
Typical configuration file
--------------------------------

The configuration file uses a section for each module of MLKAPS. Each module section must be written in all caps.

.. code-block::
	:caption: objectives

	{
	  "verbose": true,
	  "debug": false,
	  "EXPERIMENT": {
		"objectives": [
		  "accuracy",
		  "performance"
		]
	  },

The experiment section is used to describes the objectives of the experiment, as well as other high-level parameters.

NOTE: the order of the parameter is important as it must match ordering in kernel output.

.. code-block::
	:caption: kernel parameters

	  "PARAMETERS": {
		"DESIGN_PARAMETERS": {
		  "algorithm": {
			"Type": "Categorical",
			"Values": [
			  "mkl",
			  "later",
			  "split",
			  "blocked"
			]
		  }
		},
		"KERNEL_INPUTS": {
		  "matrix_size": {
			"Type": "int",
			"Values": [
			  31,
			  200
			]
		  },
		  "nbr_splits": {
			"Type": "int",
			"Values": [
			  0,
			  29
			]
		  }
		}
	  },

This section describes the parameters accepted by the kernel.
The parameters are split in two distinct categories: **design parameters** and **kernel inputs**.

* A **kernel input** is a parameter that is set by the user/system, and is not left to the optimizer to decide.
* A **design parameter** is an optimization knob than can be set by the optimizer to optimize the kernel.

In this example, the *algorithm parameter* is a design parameter, and the *matrix_size* and *nbr_splits* are kernel inputs.
You can see that each parameter has a type, and a range of values. The type can be either **int**, **float**, **Boolean**, or **Categorical**.

.. code-block::
	:caption: Kernel sampling

	  "SAMPLING": {
		"sampling_verbose": true,
		"sampling_debug": false,
		"sampler": {
		  "sampling_method": "lhs"
		},
		"sample_count": 3200,
		"scripts": {
		  "kernel": "ml-kaps_run.sh"
		},
		"kernel_arguments_order": [
		  "matrix_size",
		  "nbr_splits",
		  "algorithm"
		]
	  },

The kernel sampling defines how MLKAPS will run the kernel script. It includes the types of sampler to use,
as well as the order of the arguments to run the kernel.

NOTE: the path to the kernel script is relative to the configuration file, and not the current working directory. We advise to use absolute paths.

.. code-block::
	:caption: Modeling

	  "MODELING": {
		"modeling_verbose": true,
		"modeling_debug": false,
		"modeling_method": "xgboost",
		"xgboost_regressor_parameters": {
		  "n_estimators": 100,
		  "max_depth": 10,
		  "max_leaves": 0,
		  "n_jobs": -1
		},
		"automl_regressor_parameters": {
		  "time_left_for_this_task": 30,
		  "n_jobs": -1
		}
	  },

The modeling phase defines the parameters used for the surrogates models. Currently, MLKAPS propose
Lightgbm and XGBoost.

The `modeling_method` field defines which model to use for the main surrogate.

.. code-block::
	:caption: Optimization

	  "OPTIMIZATION": {
		"optimization_verbose": true,
		"optimization_debug": false,
		"sampling": {
		  "sampler": {
			"sampling_method": "grid"
		  },
		  "sample_count": {
			"matrix_size": 85,
			"nbr_splits": 16
		  }
		},
		"optimization_method": "genetic",
		"optimization_parameters": {
		  "evolution": {
			"pop_size": 80,
			"n_offsprings": 20
		  },
		  "termination": {
			"time": "00:00:10"
		  },
		  "selection_method": "normalized",
		  "selection_parameters": {
			"coefficients": {
			  "accuracy": 0.5,
			  "performance": 0.5
			}
		  }
		}
	  },

The optimization phase defines the parameters used for the optimization algorithm. Currently, MLKAPS only supports
 NSGAII genetic algorithms, but other algorithms may be added in the future.

It also includes a sampler that is used to generate the input configurations that will be run through the optimizer.

.. code-block::
	:caption: Clustering

	  "CLUSTERING": {
		"clustering_verbose": true,
		"clustering_debug": false,
		"clustering_method": "decision_tree",
		"clustering_parameters": {
		  "max_depth": 30
		}
	  }
	}

The clustering section defines which method to use for the clustering algorithm, as well as its parameters.

When the goal is to generate decision tree code one must use the decision tree algorithm. For decision tree the `max_depth` parameters is to be set to a value high enough to ensure that the tree captures the whole optimization space, but small
enough that it doesn't impedes the performance of the kernel.
