Writing a benchmark for MLKAPS
==================================

MLKAPS requires the user to write a benchmark that takes input parameters and design parameters as argument and return the value(s) of the objective(s) to optimize.

-------------------
Extract your kernel
-------------------

If your kernel is part of a larger application or library, we highly recommend (though it is not mandatory)
you extract the kernel and its dependencies into a separate project to minimize sampling cost.


----------------
Defining objectives
----------------

Next, you should define the objectives you want to study.
For example, add timers around the kernel if you want to study its performance.
Note that objectives are considered to be floats.


**MLKAPS considers that the smaller the objective, the better.** To maximize an objective one can negate or take the inverse.

Printing the objectives
-----------------------

Finally, those metrics **must be the last output to the terminal, formatted as a comma-separated list of values**.

.. code-block::
	:caption: Expected terminal output

	<any output>
	objective_1, objective_2, ..., objective_n
	<end of output>

Metrics tuning
-----------------------

Good metrics are the first step towards good results with MLKAPS.
If your kernel metrics consists of low/high values,
you should consider using a logarithmic/exponential scale to scale the differences between them.
This will allow MLKAPS to better explore the search space.


Another example would be to normalize the metrics by the size of the input, if it is one of the parameters of your kernel.
