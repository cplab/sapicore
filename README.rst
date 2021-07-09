Sapicore
========

A PyTorch-based modeling framework for neuromorphic olfaction.

Sapicore is a framework which provides a high level abstractions for writing
neuromorphic models using pytorch. Sapicore does not contain any concrete models,
instead each model should have its own repo that implements the Sapicore
components used by the model.

For example, the EPL-feed-forward model implemented using Sapicore can be found
`here <https://github.com/cplab/sapinet_eplff>`_. Following this methodology
will allow Sapicore to be used by multiple models independently without
each polluting the other with different implementation details or
requirements.

Models Sapicore common to a lab can be placed in a package outside the
framework and re-used by other projects that want to use these common models.

Installation
------------

Sapicore has minimal requirements. It requires

* Python 3.7+
* Pytorch 1.5+ (see `PyTorch installation <https://pytorch.org/get-started/locally/>`_).
* Scientific stack (see ``setup.py``) and tensorboard and tensorboardx (optional).

  The easiest way is to install them with conda as follows::

      conda install -c conda-forge numpy tqdm pandas ruamel.yaml tensorboard tensorboardx

  or using pip, simply::

      python -m pip install tensorboard tensorboardx

Once the dependencies are installed, to install Sapicore in the current
conda/pip environment:

* Clone sapicore::

      git clone https://github.com/cplab/sapicore.git
* `cd` into sapicore::

      cd sapicore
* Install it as an editable install::

      pip install -e .

Example model
-------------

Authors
-------

- Neuromorphic algorithms by Ayon Borthakur.
- Chen Yang.
- Framework architecture by Matthew Einhorn.
