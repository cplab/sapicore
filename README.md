Sapicore
========

A Framework for Neuromorphic Modeling
-------------------------------------

Sapicore is a spiking neural network (SNN) simulator built with PyTorch. It streamlines
the design and testing of heterogeneous model architectures with nontrivial dynamics.

We provide programmatic and dictionary/YAML-based APIs for specifying model architectures,
running simulation pipelines, and performing rudimentary output data exploration.
Sapicore's incremental class hierarchy includes efficient default implementations of neuron
and synapse models commonly used in neuroscience and machine learning applications.
Users may write their own dataloaders, derivative classes, and specification files in separate repositories.

Sapicore interfaces with industry-standard ML libraries (e.g.,
[ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html),
[scikit-learn](https://scikit-learn.org/stable/), [networkX](https://networkx.org/), and
[tensorboard](https://pytorch.org/docs/stable/tensorboard.html)).
We also support object configuration using [tree-config](https://github.com/matham/tree-config/).

***

Current Release
---------------
Sapicore 0.3.0.dev0 is currently in beta. The current version includes the following features:

* Spiking neurons (LIF and IZ).
* Analog neurons and oscillators.
* Static and STDP synapses.
* Network construction and simulation tools.
* Configurable pipelines.
* Visualization tools.

To simulate a network from a YAML configuration, simply run:

    python simulation.py -config /path/to/your/filename.yaml

See the `tutorials` directory and configuration files under `tests/engine/network/test_network`
for instructive examples.

***

Installation
------------
General requirements (see ``setup.py`` for more information):

* Python 3.10+
* Scientific Stack
* PyTorch 1.12.0+
* NetworkX 2.8.8+
* Tensorboard 2.9.0+

For developers:

* Pytest 7.1.2+
* Sphinx 5.3.0+

To install the latest stable version of Sapicore:

	pip install sapicore

To install the most recent development version:

	pip install https://github.com/cplab/sapicore/archive/refs/heads/main.zip

To run tests, change directory to the one containing `pytest.ini` (currently `sapicore`) and call:

    pytest -v -s

To run tests with a coverage report, run `sapicore/tests/scripts/run_tests.py`.
The HTML will be generated in a separate directory on the same level (open `index.html`).

***

Development
-----------
If you would like to modify or extend this library:

* Clone the repository:

      git clone https://github.com/cplab/sapicore.git

* Create a conda virtual environment (optional):

      conda create -n <env_name> python=3.10
      conda activate <env_name>

* Change directory to `sapicore` and install with pip:

      cd sapicore
      pip install -e .

This project uses ``black`` to format code and ``flake8`` for linting. We support ``pre-commit``.
To configure your local environment, install these development dependencies and set up the commit hooks:

	pip install black flake8 pre-commit
	pre-commit install

Documentation can be compiled by installing Sphinx and RTD, then running `docs/refresh.sh`.

Citation
--------
Sapinet, the primary focus of this effort, is a multilayer spiking model designed for few-shot online learning of
multiple inputs without catastrophic forgetting and without the need for data-specific hyperparameter
retuning. Key features of Sapinet include denoising, regularization, scaling, classification, and stimulus
similarity mapping.

If you use Sapicore, please cite the following article:

* Borthakur, A. (2022). [Sapinet: A sparse event-based spatiotemporal oscillator for learning in the
wild](https://arxiv.org/abs/2204.06216). <i>arXiv:2204.06216</i>.

If you are interested in dynamical perspectives on neural computation, oscillatory synchronization,
or top-down control over representational state transitions, check out and cite the following articles:

* Moyal, R. & Edelman, S. (2019). [Dynamic Computation in Visual Thalamocortical
Networks](https://www.mdpi.com/1099-4300/21/5/500). <i>Entropy, 21</i>(5).


* Edelman, S. & Moyal, R. (2017). [Fundamental computational constraints on the time course of perception and
action](https://www.sciencedirect.com/science/article/abs/pii/S007961231730050X).
<i>Progress in Brain Research, 236</i>, 121-141.

Authors
-------
A project of the [Computational Physiology Laboratory](https://cplab.net/) at Cornell University.

- Framework architecture by Matthew Einhorn and Roy Moyal.
- Neuromorphic algorithms by Roy Moyal, Ayon Borthakur, and Thomas Cleland.
- Engine infrastructure, simulator, and visualization tools by Roy Moyal.
- Tutorials and examples by Roy Moyal and Jeremy Forest.
