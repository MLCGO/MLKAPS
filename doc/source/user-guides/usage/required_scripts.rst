Required scripts
==================================

------------------
Executable script
------------------

MLKAPS requires a single a script to run your kernel, called the "executable script".
This script can be written in any language, as long as it has execution permission.

This script will receive MLAKPS's input, and need to correctly setup and run the benchmark.
MLKAPS will call the executable script with the following arguments:

.. code-block::
   :caption: Input format for the executable script:

    executable_script <value_1> <value_2> ... <value_n>

--------------------------------
(Optional) compilation script
--------------------------------

It is possible to provide MLKAPS with a compilation script.
This script will be called before the executable script, and will receive any design parameter that is a compilation flag.
MLKAPS will call the compilation script with the following arguments:

.. code-block::
   :caption: Input format for the compilation script:

    compilation_script <flag_1> <flag_2> ... <flag_n>
