MLKAPS pipeline
==================================


MLKAPS contains multiple modules that compose the optimization pipeline.
Each module is designed as a standalone unit, whose results can be stored and quick-loaded for restarting the pipeline at a given step in future runs.

The pipeline is composed of the following mandatory modules:

- :ref:`Kernel sampling`, which samples the kernel space and stores the results in a file.
  This is usually the longest module to run.

- :ref:`Modeling`, where surrogate models are created from the sampling results.

- :ref:`Optimization`, where the best design parameters for the kernel are found. This is done on a per-input basis. Depending on the algorithm, buget per optimization point, and number of point this step can be costly. However using current method all point are independent and can be run in different process (nor managed by MLKAPS yet) and it can be run offline, i.e. not on the machine required for the sampling phase.

- :ref:`Clustering`, where the optimization results are combined to create a decision tree able to output the best parameters for any input configuration. This is a very inexpensive step.

The pipeline is composed of the following optional modules:

- :ref:`Modeling validation`, which check that the surrogates are accurate enough to be used in the optimization process.
  It is also responsible to warn the user if the optimization potential of the kernel is too low,
  and check that experiments produces coherent results.


.. toctree::
   :maxdepth: 1
   :caption: Pipeline component documentations:

   pipeline/kernel_sampling
   pipeline/modeling
   pipeline/optimization
   pipeline/clustering
