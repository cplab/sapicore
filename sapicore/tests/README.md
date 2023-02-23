# Unit and Integration Testing

The `tests` module contains a growing collection of documented unit and integration tests of varying complexity,
organized by package (mirroring the project directory structure).

For instance, `test_network.py` in `tests/network` iterates over configuration files in the directory (those
specified under tests/conf/suite.yaml or all by default), creating heterogeneous connected ensembles and
logging their responses to variable current injection patterns.

By default, a directory bearing the name of the test script on the same level is used as the test root.
Some test scripts may accept runtime arguments specifying a different root directory, the path to a config file,
and/or the path to required data files (see also `projects/README.md`).

* Trivial unit tests do not reference YAMLs or data and are not configurable.

* More complex unit/integration tests may utilize pytest.mark.parameterize to quickly define a sequence of inputs
corresponding to named tests (e.g., test_network[LIF.BASIC]).

* Functionality tests are parameterized and may load YAML-specified dictionaries from disk.
Generally, those are meant to help the end user design and test complex edge cases with simple text editing.

## Requirements
To run all tests (selective for YAMLs specified in `tests/conf/suite.yaml`, where applicable),
navigate to the `pytest.ini` root directory (`sapinet`) and run `pytest -s -v`.

For a coverage report, run either `python tests/scripts/run_tests.py` or `coverage run -m pytest -s -v`
from the same directory.
