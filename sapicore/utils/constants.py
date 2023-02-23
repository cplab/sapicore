""" Constants to be used throughout the project. """
import os


CHUNK_SIZE = 10.0
""" Data chunks will be dumped to disk after reaching this size in MB. """

DT = 1.0
""" Global simulation time step (milliseconds). """

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
""" Project root source directory (.../sapicore). """

SEED = 9846
""" Random number generation seed value to be used. Invoke :func:`~utils.seed.set_seed` to fix the value
across all relevant libraries, e.g. numpy and scipy, before running a simulation. """

TIME_FORMAT = "%Y%m%d-%H%M_%S"
""" Timestamp format for run directory I/O operations. """
