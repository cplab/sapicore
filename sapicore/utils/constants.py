""" Constants to be used throughout the project. """
DT = 1.0
""" Global simulation time step (milliseconds). """

PLOT_BACKEND = "Agg"
""" Default matplotlib backend. """

SEED = 154
""" Random number generation (RNG) seed value to be used. Invoke :meth:`~utils.seed.set_seed` to fix the value
across all relevant libraries, e.g. numpy and scipy, before running a simulation. """

SYNAPSE_SPLITTERS = ["--", "->"]
""" Valid splitting characters for synapse files and name fields. For example, "L1->L2" and "L1--L2" are
allowed by default, to accommodate Windows' prohibition of the character ">" in filenames. """

TIME_FORMAT = "%Y%m%d-%H%M_%S"
""" Timestamp format for run directory I/O operations. """
