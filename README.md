Sapicore
========

A project of the [Computational Physiology Laboratory](https://cplab.net/) at Cornell University.

A Framework for Spiking Neural Network Modeling
-----------------------------------------------

Sapicore is a spiking neural network (SNN) simulator built with PyTorch. It streamlines
the design and iterative testing of neuroscience-inspired models with nontrivial dynamics.

We provide programmatic and YAML-based APIs for specifying network structure and behavior,
designing simulation pipelines, utilizing models for classification and clustering, and data visualization.

Sapicore's incremental class hierarchy includes efficient default implementations of neuron
and synapse models commonly used in neuroscience and machine learning applications.
Users may write their own dataloaders, derivative classes, and specification files in separate repositories.

Sapicore interfaces with industry-standard ML libraries, including
[ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html),
[scikit-learn](https://scikit-learn.org/stable/), [networkX](https://networkx.org/), and
[tensorboard](https://pytorch.org/docs/stable/tensorboard.html).
Object configuration using [tree-config](https://github.com/matham/tree-config/) is also supported.

***

Current Release
---------------
Sapicore 0.3.0 is in beta. The current version includes the following features:

* Spiking neurons (LIF and IZ).
* Analog neurons and oscillators.
* Static and STDP synapses.
* Network design and simulation tools.
* Scikit-compatible model API (fit/predict).
* Sampling and cross validation tools.
* Data classes with basic ETL support.
* Visualization tools.

To simulate a network from a YAML configuration using the default simulation pipeline:

    python simple.py -config /path/to/config_file.yaml

See `tutorials` and `tests/engine/network/test_network` for instructive scripts and YAML files.

***

Installation
------------
Basic requirements include Python 3.10+, PyTorch 1.12+, NetworkX, and the scientific stack (numpy, scipy, pandas).
Developers should also install pytest and sphinx.

### Install Sapicore:

* Via poetry (after installing [poetry](https://python-poetry.org/))

      git clone git@github.com:cplab/sapicore.git
      cd sapicore/
      poetry install
      poetry shell

**NOTE**: the command `poetry shell` should be used to spawn the virtual environment every time you use a new console. For reference see: https://python-poetry.org/docs/cli/#shell


* Pip (current development)

      pip install git+https://github.com/cplab/sapicore.git

* Via pypi (latest release - not currently possible)

      pip install sapicore

* Via conda

      git clone git@github.com:cplab/sapicore.git
      cd sapinet2/
      conda create -n sapinet python=3.10
      conda activate sapicore
      conda env update --file environment.yml


### Verify your Sapinet 2 install by running the tests:
* Via poetry:

      poetry install -E 'test-only'
      poetry run pytest sapinet/

* Via pip:

      pip install pytest hypothesis
      pytest sapinet/

* Via conda

      conda install pytest hypothesis
      pytest sapinet/
***
Development
-----------
Dependency management and packaging is done using ``poetry`` or ``conda`` (optional). Please install your prefered packaging manager as recommended in their respective documentation.

If you would like to modify or extend this library:

1. Clone the repository:

      ```
      git clone https://github.com/cplab/sapicore.git
      ```

2. Install dependencies via ``conda`` to manage the virtual environment (if you are not using ``conda``, go to step 3 directly):

* Create a conda virtual environment:

      conda create -n sapicore python=3.10

* Activate the previously created virtual environement:

      conda activate sapicore

* Install the dependencies both from the standard `environment.yml` and the `environment-dev.yml` file

      conda env update --file environment.yml
      conda env update --file environment-dev.yml

3. Install dependencies via ``poetry``. If you ran step 2 (conda install) this is optional and you can skip to step 4 :

* Install the dependencies:

      poetry install -E "dev"

* Activate the virtual environement:

      poetry shell

4. Verify your installation by running sapinet's tests. Change directory to the one containing `pytest.ini` (currently `sapicore/`) and call:

      ```
      pytest -v -s
      ```

      To run tests with a coverage report, run `sapinet/tests/scripts/run_tests.py`.
      The HTML will be generated in a separate directory on the same level (open `index.html`).


5. This project uses ``black`` to format code and ``flake8`` for linting. We make use of ``pre-commit`` to ensure code standards before any commit is being made.

	```
      pre-commit install
      ```

If you end up implementing additional features, feel free to submit a pull request (PR).
Feature and bug fix PRs must include functional, minimally tested code that produces the expected outcome.
Test, documentation, and proposal PRs should be substantial.


The development environemnt include all the package required to compile the documentation, which can be done by running `docs/refresh.sh`.

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
[Roy Moyal](https://scholar.google.com/citations?user=P8Ztxr4AAAAJ),
[Matthew Einhorn](https://matham.dev/about/), [Jeremy Forest](https://jeremyforest.netlify.app/),
[Ayon Borthakur](https://borthakurayon.github.io/), [Thomas Cleland](https://cplab.net/people/thomas-cleland/).

- Engine, simulator, pipelines, data classes, and visualization tools by Roy Moyal.
- Framework architecture by Matthew Einhorn and Roy Moyal.
- Algorithms by Roy Moyal, Ayon Borthakur, and Thomas Cleland.
- Tutorials, utilities, and examples by Roy Moyal and Jeremy Forest.
