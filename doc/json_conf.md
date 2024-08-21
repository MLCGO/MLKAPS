# JSON configuration file

The JSON configuration file allows the user to specify the different parameters of his experiments and to customize the
different ML-Kaps steps.

This file consists of 7 sections. Each one containing required and optional fields. The sections are:

* [General parameters](#general-parameters)
* [PARAMS](#params)
* [DATA_COLLECTION](#data_collection)
* [DATA_PREPROCESSING](#data_preprocessing)
* [MODELING](#modeling)
* [OPTIMIZATION](#optimization)
* [CLUSTERING](#clustering)

An [example](experiments/BF16x3/config.json) is available in the BF16x3 experiement delivered with this sofware.

# General Parameters

This includes :

* `"experiment_folder"` (optional): path to experiment folder. ML-Kaps output are redirected to this folder. If not
  specified, `"experiment_folder"` is the folder containing the JSON configuration file. If the specified experiment
  folder does not exist, it is created.
* `"verbose"` (optional): global verbose parameter. If `true`, ML-Kaps prints a more detailled output. This parameter is
  inherited by all ML-Kaps steps, unless they have their own verbose specified. Default value is `false`.
* `"debug"` (optional): global debug parameter. If `true`, ML-Kaps prints debug statements allowing the user to track
  the ML-Kaps steps to potentially detect errors. This parameter is inherited by all ML-Kaps steps, unless they have
  their own debug specified. Default value is `false`.

# PARAMS

In this section, the user specifies the different user and kernel parameters, their names, types and ranges.

* `"DESIGN_PARAMETERS"` (required): subsection containing details on the kernel design parameters.
    * Add a subsection `"specified_kernel_param"` (required) for each kernel parameter
        * `Type` (required): variable type in `Categorical`, `int`, `float`, `boolean`
        * `Values` (required): For Categorical variables, define the list of all possbile string values.
          Example : `["1sub","2sublow","2subhigh","3sub"]`. For boolean variables, define *[false,true]*. For integer
          and float variables, define an interval with the minimum and maximum value. Example : `[1,1000]`.
* `"KERNEL_INPUTS"` (required): subsection containing details on the user parameters.
    * The subsections are defined in a similar way as `DESIGN_PARAMETERS`

# DATA_COLLECTION

* `"data_collect_verbose"` (optional): verbose for the data collection step. If specified, it overwrites the verbose
  specified in the general parameters.
* `"data_collect_debug"` (optional): debug for the data collection step. If specified, it overwrites the debug specified
  in the general parameters.
* `"pre_exec"` (optional): bash script to execute before running the executable. Example:  `"setvars.sh"`.
* `"exec_file"` (required): kernel executable name. This executable should print either 1 floating point if there is
  only one objective to optimize, or print 2 floating points separated by a comma if there are 2 objectives to optimize.
  Example : `"a.out"`.
* `"exec_arguments_order"` (required): array containing the arguments for provided executable in order.
  Example : `["summation","alternate","block_size","cond","vec_size"]`
* `"OBJECTIVES"` (required): array of the different objectives to optimize. Must correspond to the output of the
  provided executable. Example : `["accuracy","performance"]`
* `"NB_PERF_MEASURES"` (optional): Number of objectives measures before taking the median. Set to `5` by default.
* `"data_collect_method"` (required): Sampling method. Supported methods are : `"grid"` (exhaustive grid sampling)
  , `"lhs"` (latin hypercube sampling), and `"random"` (random sampling).
* '"data_collect_method_params"' (required): Sampling parameters.
    * If the `"grid"` sampling method has been chosen, we define a subsection `"param"` (required) for each integer or
      floating point user and kernel parameter.
        * `"nb_values"` (optional): number of values to sample from this parameter within the specified range. Set
          to `100` by default.
        * `"Scale"` (optional): `"lin"` or `"log"` to choose either a linear-scale or log-scale sampling within the
          specified range. Set to `"lin"` by default.
    * If the `"lhs"` or `"random"` sampling has been specified, we define a subsection `"num_samples"` (optional) that
      defines the number of values to sample in the whole parameters space. Set to 100 by default.
* `"DATA_COLLECT_FILE"` (optional): csv file name containing the input parameters and their corresponding measured
  objectives. Set to `"data_collect.csv"` by default.
* `"visualization"` (optional): section that defines the data collection visualization parameters.
    * Add a subsection `"user_param"` (optional) for each user_param.
        * `"Scale"` (optional): `"lin"` or `"log"`, to specify the image axis scale. Set to `"lin"` by default.
    * `data_collect_path_prefix"` (optional): visualization output path prefix for saving. If not specified, images are
      saved in the experiment folder.
    * `"data_collect_plot_prefix"` (optional): visualization name prefix for saving. If not specified, images have a
      default name related to the current step and experiment.
    * `"dpi"` (optional): visualization output dpi.
      Equal to `100` by default.

# DATA_PREPROCESSING

* `"data_preprocessing_verbose"` (optional): verbose for the data preprocessing step. If specified, it overwrites the
  verbose specified in the general parameters.
* `"data_preprocessing_debug"` (optional): debug for the data preprocessing step. If specified, it overwrites the debug
  specified in the general parameters.
* `"TRANSFORM"` (optional): subsection containing the function to apply to each objective
    * `"objective_name"` (optional, replace objective_name by each objective you want to apply a function on) :
      transformation to apply. Supported transformations include `"apply_log"`,`"apply_negative_log"`, and `"apply_exp"`
      .
* `"CUSTOM_DISTR"` (optional): subsection containing the parameters of the custom distribution multiplication
  preprocessing.
    * `"PARAM"` (required): specify the distribution parameter.
    * `"DISTR_TYPE"` (required): distribution name. Supported distribution is `"gaussian"`.
    * `"MEAN"` (required): distribution mean.
    * `"STD"` (required): distribution standard deviation.
* `"visualization"` (optional): data_preprocessing visualization parameters.
    * `"user_param"` (optional) for each user_param.
        * `"Scale"` (optional): `"lin"` or `"log"`, to specify the image axis scale. Set to `"lin"` by default.
    * `"preprocessing_path_prefix"` (optional): visualization output path prefix for saving. If not specified, images
      are saved in the experiment folder.
    * `"preprocessing_plot_prefix"` (optional): visualization name prefix for saving. If not specified, images have a
      default name related to the current step and experiment.
    * `"dpi"` (optional): visualization output dpi.
      Set to `100` by default.

# MODELING

* `"modeling_verbose"` (optional): verbose for the modeling step. If specified, it overwrites the verbose specified in
  the general parameters.
* `"modeling_debug"` (optional): debug for the modeling step. If specified, it overwrites the debug specified in the
  general parameters.
* `"modeling_method"` (required): Model name. Supported models include : `"automl"`,`"xgboost"`,`"elastic_net"`
  ,`"random_forest"`
* `"model_params"` (optional): Model parameters written in a dictionary format. Example for automl modeling :

```
"model_params" : {
   "time_left_for_this_task": 60,
   "seed": 5
}
```

* `"visualization"` (optional): visualization parameters for the modeling section
    * `"user_param"` (optional) for each user_param.
        * `"Scale"` (optional): `"lin"` or `"log"`, to specify the image axis scale. Set to `"lin"` by default.
    * `"modeling_path_prefix"` (optional): visualization output path prefix for saving. If not specified, images are
      saved in the experiment folder.
    * `"modeling_plot_prefix"` (optional): visualization name prefix for saving. If not specified, images have a default
      name related to the current step and experiment.
    * `"dpi"` (optional): visualization output dpi.
      Set to `100` by default.

# OPTIMIZATION

* `"optimization_verbose"` (optional): verbose for the optimization step. If specified, it overwrites the verbose
  specified in the general parameters.
* `"optimization_debug"` (optional): debug for the optimization step. If specified, it overwrites the debug specified in
  the general parameters.
* `"optimization_sampling_method"` (required): Sampling method used to generate some user points.The optimization
  process is then run on each of these points. Supported optimization sampling method include `"grid"`.
* `"optimization_sampling_parameters"` (optional): user points sampling parameters.
    * If the `"grid"` sampling has been specified, add a subsection `"user_param"` (optional) for each user parameter.
        * `"Nb_values"` (optional): number of values to sample for the specified user parameter within the corresponding
          range. Set to `100` by default.
        * `"Scale"` (optional): `"lin"` or `"log"`. Specify the scale of the user parameter sampling. Equals to `"lin"`
          by default.

* `"otimization_method"` (required): Optimization method used to find the best kernel parameters for a given user
  point.Supported optimization method include `"genetic"`.

* `"optimization_method_params"` (required): Parameters for the optimization algorithm.
    * For genetic optimization, add an `"evolution"` (optional) subsection containing the paramaters of the genetic
      algorithm parameters.
        * `"pop_size"` (optional): genetic algorithm population size.
        * `"n_offsprings"` (optional): number of generated offsprings at each generation.

    * `"termination"` (required): genetic algorithm termination criteria. Add one of the following subsections:
        * `"time"`(optional): maximum time of an optimization run. For example: `"00:00:10"`.
        * `"n_gen"` (optional): number of generations before ending the algorithm. For example: `20`.

    * `"selection_method"` (optional): Selection method used to select one set of kernel design parameters from the
      final genetic population .Supported selection methods include `"naive"`,`"mean"`,`"normalized_selection"`. Set
      to `"normalized_selection"` by default.
    * `"selection_params"` (optional): Selection method parameters. For the normalized selection, the user may add the
      following subsections:
        * `"acc_coeff"` (optional): Weight to put on the first specified objective. Set to `1` by default.
        * `"perf_coeff"` (optional): Weight to put on the second specified objective. Set to `1` by default.
* `"visualization"` (optional) : visualization parameters for the optimization section
    * Add a subsection `"user_param"` (optional) for each user_param.
        * `"Scale"` (optional): `"lin"` or `"log"`, to specify the image axis scale. Set to `"lin"` by default.
    * `"optim_path_prefix"` (optional): visualization output path prefix for saving. If not specified, images are saved
      in the experiment folder.
    * `"optim_plot_prefix"` (optional): visualization name prefix for saving. If not specified, images have a default
      name related to the current step and experiment.
    * `"dpi"` (optional): visualization output dpi.
      Set to `100` by default.

# CLUSTERING

* `"clustering_verbose"` (optional): verbose for the clustering step. If specified, it overwrites the verbose specified
  in the general parameters.
* `"clustering_debug"` (optional): debug for the clustering step. If specified, it overwrites the debug specified in the
  general parameters.
* `"clustering_method"` (required): clustering model to use. Supported models include `"automl"` and `"decision_tree"`.
* `"clustering_params"` (optional): clustering models parameters, specified in the form of a dictionary. For a decision
  tree clustering model, we may for example use:

```
"clustering_params" : {
    "max_depth": 5,
    "random_state": 0
}
```

* `"visualization"` (optional): visualization parameters for the clustering section
    * `"user_param"` (optional) for each user_param.
        * `"Scale"` (optional): `"lin"` or `"log"`, to specify the image axis scale. Set to `"lin"` by default.
    * `"clustering_path_prefix"` (optional): visualization output path prefix for saving. If not specified, images are
      saved in the experiment folder.
    * `"clustering_plot_prefix"` (optional): visualization name prefix for saving. If not specified, images have a
      default name related to the current step and experiment.
    * `"dpi"` (optional): visualization output dpi.
      Set to `100` by default.
          
    
