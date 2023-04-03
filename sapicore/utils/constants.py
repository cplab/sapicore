""" Constants to be used throughout the project. """
DT = 1.0
""" Global simulation time step (milliseconds). """

SEED = 154
""" Random number generation (RNG) seed value to be used. Invoke :meth:`~utils.seed.set_seed` to fix the value
across all relevant libraries, e.g. numpy and scipy, before running a simulation. """

TIME_FORMAT = "%Y%m%d-%H%M_%S"
""" Timestamp format for run directory I/O operations. """
