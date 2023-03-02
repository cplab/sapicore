# Test Suite

A growing collection of documented unit, integration, and functional tests of varying complexity.

Tests are organized by package (mirroring the project directory structure). For instance, `test_network.py`
in `tests/network` iterates over configuration YAMLs in the directory, creating heterogeneous connected
ensembles and logging their responses to variable current injection patterns.

By default, a directory bearing the name of the test script on the same level is used as the test root.
Some test scripts may accept runtime arguments specifying a different root directory, the path to a configuration
file, and/or the path to required data files.

* Trivial unit tests do not reference YAMLs or data and are not configurable.

* More complex tests may utilize pytest.mark.parameterize to quickly define a sequence of inputs
corresponding to named tests (e.g., `test_network[LIF]`).

* Functional tests are usually parameterized and may load YAML-specified dictionaries from disk.
Generally, those are meant to help the end user design and test complex edge cases with simple text editing.

## Requirements
To run multiple tests, navigate to the `pytest.ini` root directory (`sapicore`) and run `pytest -s -v`,
potentially with `-m "not <mark>"` where `<mark>` is one of the registered marks in the INI file.

For a coverage report, run either `python tests/scripts/run_tests.py` or `coverage run -m pytest -s -v`
from the same directory.
