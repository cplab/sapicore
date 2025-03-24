Sapicore
========

A PyTorch-Based Spiking Neural Network Simulation Framework
-----------------------------------------------------------

Sapicore is a spiking neural network (SNN) simulator, built with PyTorch and designed for
neuroscience-inspired models with realistic architectures and dynamics.

We provide a simple API for:

* Data loading and transformation.
* Initializing and connecting network components.
* Extending base neurons (e.g., LIF) and synapses (e.g., STDP).
* Orchestrating experiment pipelines.
* Selectively logging intermediate results.
* Deploying neuromorphic networks as machine learning models.
* Visualizing, interpreting, and analyzing output.

<b>Users may configure networks in YAML or directly in code.</b>
Develop custom neuron, synapse, network, and data loader classes.

<b>We made component extension and customization easy.</b> Default implementations of common models used in
computational neuroscience and neuromorphic machine learning applications are provided out of the box.</b>

<b>Sapicore interfaces with industry-standard ML libraries</b>, including
[scikit-learn](https://scikit-learn.org/stable/).

A project of the [Computational Physiology Lab](https://cplab.net/) @ Cornell University.

***

Current Release
---------------
Sapicore 0.4 is runtime- and memory-optimized. This beta version includes:

* Flexible data classes (metadata-based row selection).
* Spiking neurons (LIF and IZ).
* Graded neurons (integrators, oscillators).
* Static and plastic synapses (STDP).
* Automated network construction and simulation.
* Scikit-integrated ML model API (fit/predict).
* Sampling and cross validation tools.
* Visualization tools.

To simulate a network from a YAML configuration using the default simulation pipeline:

    python simple.py -config /path/to/config_file.yaml

See `tutorials` and `tests/engine/network/test_network` for instructive scripts and YAML files.

***

Installation
------------
Basic requirements include Python 3.11+, PyTorch 2.1+, NetworkX, and the scientific stack (numpy, scipy, pandas).

To install the most recent development version:

	pip install https://github.com/cplab/sapicore/archive/refs/heads/main.zip

To run tests, change directory to the one containing `pytest.ini` (`sapicore`) and call:

    pytest -v -s

To run tests with a coverage report, run `sapicore/tests/scripts/run_tests.py`.
The coverage report will be generated in a separate directory on the same level (open `index.html`).

***

Development
-----------
If you would like to modify or extend this library:

* Clone the repository:

      git clone https://github.com/cplab/sapicore.git

* Create a conda virtual environment (optional):

      conda create -n <env_name> python=3.11
      conda activate <env_name>

* Change directory to `sapicore` and install with pip:

      cd sapicore
      pip install -e .

This project uses ``black`` to format code and ``flake8`` for linting. We support ``pre-commit``.
To configure your local environment, install these development dependencies and set up the commit hooks:

	pip install black flake8 pre-commit
	pre-commit install

Documentation can be compiled by installing Sphinx and RTD, then running `docs/refresh.sh`.
See ``setup.py`` for more information.

Citation
--------
If you use Sapicore, please cite it as:

* Moyal, R., Einhorn, M., Borthakur, A., & Cleland, T. (2024). Sapicore (Version 0.4.0) [Computer software]. https://github.com/cplab/sapicore


References
----------
For more information about past and ongoing projects utilizing Sapicore, refer to the following publications:

* R. Moyal, K. R. Mama, M. Einhorn, A. Borthakur, and T. A. Cleland (2024). [Heterogeneous quantization regularizes spiking
neural network activity](https://doi.org/10.48550/arXiv.2409.18396). <i>arXiv:2409.18396</i>.

* A. Borthakur (2022). [Sapinet: A sparse event-based spatiotemporal oscillator for learning in the
wild](https://arxiv.org/abs/2204.06216). <i>arXiv:2204.06216</i>.

For a dynamical systems perspective on neural computation, temporal coding, and top-down control of
sensory processing, the following article may be of interest:

* R. Moyal and S. Edelman (2019). [Dynamic computation in visual thalamocortical
networks](https://www.mdpi.com/1099-4300/21/5/500). <i>Entropy, 21</i>(5).

Contributors
------------
[Roy Moyal](https://scholar.google.com/citations?user=P8Ztxr4AAAAJ),
[Matthew Einhorn](https://matham.dev/about/),
[Ayon Borthakur](https://borthakurayon.github.io/),
[Thomas Cleland](https://cplab.net/people/thomas-cleland/).
