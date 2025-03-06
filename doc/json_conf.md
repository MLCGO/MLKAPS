# JSON configuration file

The JSON configuration file allows the user to specify the different parameters of their experiments and to customize the different ML-Kaps steps.

This file consists of several sections. Each one containing required and optional fields. The sections are:

* [General parameters](#general-parameters)
* [OBJECTIVES](#objectives)
* [PARAMETERS](#parameters)
* [SAMPLING](#sampling)
* [MODELING](#modeling)
* [OPTIMIZATION](#optimization)
* [CLUSTERING](#clustering)

Examples are available in the [synthetic2d](../examples/synthetic2D/ga_adaptive.json) example or [openblas](../examples/openblas/config_mlkaps_template.json) delivered with this software.

This format will evolve with newer methoods avaible in MLKAPS. However we are trying as much as possible to support backward compatibility, but cannot guarantee it.

# General Parameters

This includes:

* `"experiment_folder"` (optional): path to experiment folder. ML-Kaps output is redirected to this folder. If not specified, `"experiment_folder"` is the folder containing the JSON configuration file. If the specified experiment folder does not exist, it is created.
* `"verbose"` (optional): global verbose parameter. If `true`, ML-Kaps prints a more detailed output. This parameter is inherited by all ML-Kaps steps, unless they have their own verbose specified. Default value is `false`.
* `"debug"` (optional): global debug parameter. If `true`, ML-Kaps prints debug statements allowing the user to track the ML-Kaps steps to potentially detect errors. This parameter is inherited by all ML-Kaps steps, unless they have their own debug specified. Default value is `false`. Debug can be set independently in each step of the workflow.

# OBJECTIVES

In this section, the user specifies the objectives for the experiment.

* `"objectives"` (required): subsection containing details on the objectives.
    * Add a subsection for each objective:
        * `direction` (required): direction of the objective, either `"minimize"` or `"maximize"`.
        * `bound` (optional): bound for the objective. Example: `3`.

# PARAMETERS

In this section, the user specifies the different inputs and design parameters, their names, types, and ranges.

* `"KERNEL_INPUTS"` (required): subsection containing details on the user parameters.
    * Add a subsection for each user parameter:
        * `Type` (required): variable type in `Categorical`, `int`, `float`, `boolean`
        * `Values` (required): For Categorical variables, define the list of all possible string values. Example: `["one", "two", "three"]`. For boolean variables, define `[false, true]`. For integer and float variables, define an interval with the minimum and maximum value. Example: `[-2, 2]`.
* `"DESIGN_PARAMETERS"` (required): subsection containing details on the kernel design parameters.
    * Add a subsection for each kernel parameter:
        * `Type` (required): variable type in `Categorical`, `int`, `float`, `boolean`
        * `Values` (required): For Categorical variables, define the list of all possible string values. Example: `["one", "two", "three"]`. For boolean variables, define `[false, true]`. For integer and float variables, define an interval with the minimum and maximum value. Example: `[1, 5]`.

# SAMPLING

* `"runner"` (required): type of runner, e.g., `"executable"`.
* `"runner_parameters"` (required): parameters for the runner.
    * `"kernel"` (required): path to the kernel executable or script.
    * `"timeout"` (optional): timeout for each run in seconds.
    * `"parameters_order"` (required): array containing the arguments for the provided executable in order. Example: `["x", "y", "b", "c"]`
* `"sampler"` (required): sampling method, e.g., `"ga_adaptive"`.

## ga-adaptive sampler parameters

* `"sampler_parameters"` (required): parameters for the sampler.
    * `"n_samples"` (required): number of samples.
    * `"n_iterations"` (required): number of iterations.
    * `"bootstrap_ratio"` (optional): bootstrap ratio.
    * `"initial_ga_ratio"` (optional): initial genetic algorithm ratio.
    * `"final_ga_ratio"` (optional): final genetic algorithm ratio.

# MODELING

* `"modeling_method"` (required): modeling method, e.g., `"lightgbm"`.

## Lightgbm modeling parameters
* `"parameters"` (required): parameters for the modeling method. They will depend on the chosen modeling method. Currently only lightgbm has been extensively tested.
    * `"objective"` (required): objective function, e.g., `"mae"`.
    * `"n_jobs"` (optional): number of jobs to run in parallel.
    * `"verbose"` (optional): verbosity level.
    * `"n_estimators"` (optional): number of boosting stages.
    * `"min_data_in_leaf"` (optional): minimum number of data in one leaf.
    * `"boosting"` (optional): boosting type, e.g., `"gbdt"`.
    * `"learning_rate"` (optional): learning rate.

# OPTIMIZATION

* `"sampling"` (required): sampling configuration.
    * `"sampler"` (required): sampling method, e.g., `"grid"`.
    * `"sample_count"` (required): sample count for each parameter.
* `"optimization_method"` (required): optimization method, e.g., `"genetic"`.
* `"optimization_parameters"` (required): parameters for the optimization method.
    * `"evolution"` (required): evolution parameters.
        * `"pop_size"` (required): population size.
    * `"termination"` (required): termination criteria.
        * `"time"` (required): maximum time for optimization.
    * `"selection_method"` (optional): selection method, e.g., `"mono"`.
    * `"early_stopping"` (optional): whether to use early stopping.

# CLUSTERING

* `"clustering_method"` (required): clustering method, e.g., `"decision_tree"`.