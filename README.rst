Sapicore
========

A project of the Computational Physiology Laboratory at Cornell University.

Please see the online `docs <https://cplab.github.io/sapicore/index.html>`_
for complete documentation.

-----------------

A PyTorch-based framework for neuromorphic modeling.

Sapicore is a framework that provides high level abstractions to help write
neuromorphic models using pytorch. Sapicore itself does not contain any concrete
models, instead each model may have its own repo that implements the Sapicore
components used by the model.

Following this methodology will allow Sapicore to be used by multiple models
independently, without each polluting the other with different implementation
details or requirements.

Sapicore models common to a lab can be placed in a package outside the
framework and re-used by other projects that want to use these common models.

Sapicore supports model user-configuration using the
`tree-config <https://github.com/matham/tree-config/>`_ package.
Similarly, Sapicore supports annotating properties and buffers for logging
to e.g. ``tensorboardx`` for live display or using
`nixio <https://github.com/G-Node/nixpy>`_ HDF5 based files for later analysis or
debugging.

Installation
------------

Sapicore has minimal requirements. It requires

* Python 3.7+
* Pytorch 1.5+ (see `PyTorch installation <https://pytorch.org/get-started/locally/>`_).
* Scientific stack (see the list in ``setup.py``).
* Tensorboard and tensorboardx (optional).

  The easiest way is to install them with conda as follows::

      conda install -c conda-forge numpy tqdm pandas ruamel.yaml tensorboard tensorboardx

  or using pip, simply (pip automatically installs the remaining dependencies)::

      python -m pip install tensorboard tensorboardx

Once the dependencies are installed, to install Sapicore in the current
conda/pip environment:

User install
************

You can install the latest stable Sapicore with::

    pip install sapicore

To install the latest Sapicore from github, do::

    pip install https://github.com/cplab/sapicore/archive/refs/heads/main.zip

Development install
*******************

To install Sapicore for development and editing Sapicore itself:

* Clone sapicore from github::

      git clone https://github.com/cplab/sapicore.git

* `cd` into sapicore::

      cd sapicore

* Install it as an editable install::

      pip install -e .

Example model
-------------

Following is brief runnable example. A similar but complete example with configuration and logging
can be found under sapicore/examples.

.. code-block:: python

    import torch
    from sapicore.model import SapicoreModel
    from sapicore.neuron.analog import AnalogNeuron
    from sapicore.synapse import SapicoreSynapse
    from sapicore.pipeline import PipelineBase
    from sapicore.learning import SapicoreLearning


    class SimpleNeuron(AnalogNeuron):
        """Represents a neuron or matrix of neurons."""

        activation: torch.Tensor

        def forward(self, data: torch.tensor) -> torch.tensor:
            self.activation = torch.clip(data, -2, 2)
            return self.activation


    class SimpleSynapse(SapicoreSynapse):
        """Represents a synapse connecting a neuron or matrix of neurons."""

        weight: torch.Tensor

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # register weight as part of the state so it is saved with the model
            self.register_buffer("weight", torch.zeros(0))
            self.weight = torch.normal(0, 1, size=(5, ))

        def forward(self, data: torch.tensor) -> torch.tensor:
            return data * self.weight


    class SimpleLearning(SapicoreLearning):
        """Learns the synapse weight based on pre-post activation."""

        def apply_learning(
                self, pre_neuron: SimpleNeuron, synapse: SimpleSynapse,
                post_neuron: SimpleNeuron, **kwargs):
            synapse.weight *= torch.abs(pre_neuron.activation) * \
                torch.abs(post_neuron.activation)


    class MyModel(SapicoreModel):
        """Network model that contains neurons/synapses."""

        neuron_1: SimpleNeuron

        synapse: SimpleSynapse

        neuron_2: SimpleNeuron

        learning: SimpleLearning

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.neuron_1 = SimpleNeuron()
            self.synapse = SimpleSynapse()
            self.neuron_2 = SimpleNeuron()

            self.add_neuron(None, 'entry_neuron', self.neuron_1)
            self.add_synapse(self.neuron_1, 'synapse', self.synapse)
            self.add_neuron(self.synapse, 'exit_neuron', self.neuron_2)

            self.learning = SimpleLearning()
            self.add_learning_rule('learning', self.learning)

        def initialize_learning_state(self) -> None:
            self.learning.initialize_state()

        def forward(self, data: torch.tensor) -> torch.tensor:
            data = self.neuron_1(data)
            data = self.synapse(data)
            data = self.neuron_2(data)
            return data

        def apply_learning(self, **kwargs) -> None:
            self.learning.apply_learning(self.neuron_1, self.synapse, self.neuron_2)


    class SimplePipeline(PipelineBase):
        """Training pipeline."""

        def run(self) -> None:
            use_cuda = torch.cuda.is_available()
            cuda_device = torch.device("cuda:0" if use_cuda else "cpu")

            model = MyModel()
            model.initialize_state()
            model.initialize_learning_state()
            model.to(cuda_device)

            print('Pre-learning weight: ', model.synapse.weight.cpu().numpy())

            # these models don't use gradients
            with torch.no_grad():
                for i in range(3):
                    # fake data
                    data = torch.normal(0, 1, size=(5, ))
                    # pass it through the model
                    model.forward(data)
                    # apply model learning
                    model.apply_learning()

            print('Post-learning weight: ', model.synapse.weight.cpu().numpy())


    if __name__ == '__main__':
        # create and run the model
        pipeline = SimplePipeline()
        pipeline.run()

When run, this print::

    Pre-learning weight:  [-0.95982265 -0.2735969   0.6473335  -0.37592512  0.05847792]
    Post-learning weight:  [-6.0495706e-09 -8.3768668e-08  3.3906079e-05 -3.3586942e-09
      1.3144294e-32]

Authors
-------

A project of the Computational Physiology Laboratory at Cornell University.

- Neuromorphic algorithms by Ayon Borthakur and Thomas Cleland.
- Framework architecture by Matthew Einhorn.
