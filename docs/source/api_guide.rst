Usage guide
===========

Sapicore is a framework, which provides a high level abstractions for writing
neuromorphic models using pytorch. Sapicore itself does not contain any concrete
models, instead it provides abstractions that help model writers implement their
models in an organized and well defined fashion.

It also provides infrastructure to help configure and log model parameters to ease
debugging and model analysis.

It is expected that model writers will create libraries of Sapicore based components
that can be re-used by models. These concrete components and models are intentionally
not part of Sapicore.

See the ``sapicore/examples`` directory in the repo on GitHub for concrete
implementation examples.

PyTorch
-------

Sapicore is written on top of PyTorch. This means that Sapicore classes that
perform computation and may need to interact with the CPU/GPU inherit from
``torch.nn.modules.module.Module``.

These class instances are added (whether directly or through a parent) to the
overall model so that pytorch can track them and manage their :ref:`state-guide`.
Any data and parameters or buffers used by the model must similarly be sent to
the correct CPU/GPU device running the model (especially if parallelizing),
and be registered as part of the :ref:`state-guide`.

A PyTorch convention is that when a model is called (e.g. ``model(data)``),
it'll automatically call the ``forward`` method of the model (i.e.
``model.forward(data)``). This is how data should be passed to the
model/network/synapse/neuron. See the examples for a demonstration.

Model
-----

API
~~~

Network abstraction
*******************

Sapicore defines the basic abstract classes used to build neuromorphic models.

At the lowest level, we have a :py:class:`~sapicore.neuron.SapicoreNeuron` and
a :py:class:`~sapicore.synapse.SapicoreSynapse` that are composed in various ways to
build a :py:class:`~sapicore.network.SapicoreNetwork`. If a network is a graph
composed of vertices and nodes, then a neuron is a node and a synapse is a vertex.

Sapicore does not define what it means for neurons to be connected to each other
through synapses, just that they are connected. The implementation decides the
concrete details, including whether a synapse or neuron object actually represents
a cluster of synapses/neurons.

Neurons, synapses, and networks have an ``initialize_state`` method that would be called
indirectly through the :py:`class:~sapicore.model.SapicoreModel` to ready the
state. This would happen after :ref:`config-guide`. Then, their ``forward`` method
would be called to process the data through the neuron/synapse. The network has to
implement its ``forward`` method to call the appropriate neurons/synapses in the correct
order to pass its input through the network.

Model abstraction
*****************

So far, a network only defined the connectivity between neurons and synapses.
A :py:class:`~sapicore.model.SapicoreModel`, although a network itself, allows
sub-networks to be added as nodes to its connectivity graph. So, we can define
concrete networks and then re-use them in models without having to think
about their internal details, but we just pass data through their ``forward``
method as if it's a neuron.

Learning
^^^^^^^^

A :py:class:`~sapicore.model.SapicoreModel` also adds the concept of a
:py:class:`~sapicore.learning.SapicoreLearning` rule that can be added
to the model. Its behavior is similarly left undefined and must be
concretely implemented.

A :py:class:`~sapicore.learning.SapicoreLearning` rule also has an
:py:meth:`~sapicore.learning.SapicoreLearning.initialize_state` method
as well as an :py:meth:`~sapicore.learning.SapicoreLearning.apply_learning` method
that can be called with objects as arguments on which it applies the
learning. The model, in addition to a
:py:meth:`~sapicore.model.SapicoreModel.initialize_learning_state`
(that must be overwritten to call
:py:meth:`~sapicore.learning.SapicoreLearning.initialize_state` on the relevant
learning rules), also has an :py:meth:`~sapicore.model.SapicoreModel.apply_learning`
method that also needs to be implemented by the user (to call
:py:meth:`~sapicore.learning.SapicoreLearning.apply_learning`). Notice that the
model won't automatically call these methods on learning rules.

.. _state-guide:

Model state
~~~~~~~~~~~

A model requires access to data as well as memory buffers and parameters.
However, depending on the device running the model (CPU or specific GPU),
we have to be able to send the data as well as the model itself and all
its containing modules (neurons, synapses, etc.) and buffers to that device.

In PyTorch we'd do something like:

.. code-block:: python

    # use cpu or first gpu
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
    else:
        cuda_device = torch.device("cpu")

    # send it all to the device
    model.to(cuda_device)

Additionally, PyTorch supports saving all the model buffers and parameters
to a state file, and then restoring the the model to that state from the
file. E.g.:

.. code-block:: python

    # save the model state to a file
    torch.save(model.state_dict(), filename)
    # then load and restore model to it
    model.load_state_dict(torch.load(filename))

To be able to do this:

#. Sapicore inherits every learning rule, network, synapse, neuron, and other
   relevant objects from ``torch.nn.modules.module.Module``.
#. As part of the Sapicore API, objects are added directly or indirectly through a parent to
   the root model. E.g. :py:meth:`~sapicore.model.SapicoreModel.add_learning_rule`
   registers the rule object with PyTorch as a "child" of the model. This is how PyTorch
   can track all relevant objects and send them and their memory to the device.
#. For each object, if it uses any memory buffers or parameters (i.e. tensors), it must be
   manually registered by the user (see below). This is how PyTorch knows which memory to
   copy and save/restore.

With this, the complete model and its data can be saved to the device and to disk
like the above example.

Any tensors used by the model that is part of the model state, must be registered with
PyTorch. It can be registered as a buffer (:py:meth:`torch.nn.Module.register_buffer`)
or parameter (:py:meth:`torch.nn.Module.register_parameter`). A parameter is just a
buffer whose gradients is tracked for the optimizer and returned in the
parameters list. Neuromporphic models don't typically use gradients so
:py:meth:`torch.nn.Module.register_buffer` is typically used. For example:

.. code-block:: python

    class SimpleSynapse(SapicoreSynapse):

        # the synapse weight is part of the state. We don't set it here to None because
        # pytorch cannot handle buffers declared at a class level
        weight: torch.Tensor

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # register weight as part of the state and give it a value
            self.register_buffer("weight", torch.zeros(0))
            self.weight = torch.zeros(0)

        def initialize_state(self, model_size, **kwargs):
            super().initialize_state(**kwargs)
            # set the weight as needed
            self.weight = torch.normal(self.mean, self.std, size=(model_size, ))

        def forward(self, data: torch.tensor) -> torch.tensor:
            # use the weight as needed
            return data * self.weight

Pipeline
--------

Running an experiment on a model is very similar to how deep learning PyTorch models
are trained, except that we don't use back-propagation or the deep learning optimizers.

:py:class:`~sapicore.pipeline.PipelineBase` is a very simple base class to be used to run
training or testing. The typical overall steps to run an experiment as demoed in the
example is to:

#. Parse any command line options and save them as class properties.
#. Initialize PyTorch and any libraries used.
#. Load any datasets to be used. :py:class:`torch.utils.data.Dataset` is used
   to load the data in a format usable with PyTorch. The ``torchvision`` project
   has good examples on how to load datasets.
#. Create the :py:class:`~sapicore.model.SapicoreModel` to be trained/tested.
#. Configure the model as described in the :ref:`config-guide` section.
#. Set up and configure which properties to log during training/testing as described
   in the :ref:`log-guide` section.
#. Initialize the model and learning state and "send" the model to the CPU/GPU device used.
#. Run the training/testing iterations. For each iteration, use ``with torch.no_grad(): ...``
   to disable gradient tracking, pass the data through the model, and apply learning as needed.

.. _config-guide:

Configuration
-------------

Sapicore supports extensive model configuration using the :py:mod:`tree_config` package.
See the :py:mod:`tree_config` guide docs, but briefly, each of the model, neuron, synapse
etc classes inherit from :py:class:`tree_config.Configurable`, to make them
configurable by :py:mod:`tree_config`.

One uses the ``_config_props_`` property to list all the names of the properties
of the class that is configurable. :py:mod:`tree_config` can then gather a dict of these
properties, dump them to a yaml file, and then apply their values from a previous
yaml file to the model. Users can edit the yaml file to change the value of any
property, or it can be programmatically changed when it's in a dict form before it's
applied to the model.

Starting from the root model one uses ``_config_children_`` to tell :py:mod:`tree_config`
about children objects of the model, recursively, that also need to be configured.
See the ``sapicore/examples`` directory in the repo for examples.

For example, we can define this model:

.. code-block:: python

    class MyModel(SapicoreModel):

        _config_props_ = ('model_size', )

        model_size = 3

Then we can use :py:func:`~tree_config.read_config_from_object`,
:py:func:`~tree_config.read_config_from_file`, or
:py:func:`~tree_config.load_config` to get the config dict,
potentially further edit it, and then apply to the model using
:py:func:`~tree_config.apply_config`. To dump the config to a yaml
file one uses :py:func:`~tree_config.dump_config`. Or we just use
:py:func:`~tree_config.load_apply_save_config` to do it all in one step.

So, we could do:

.. code-block:: python

    from tree_config import load_apply_save_config

    model = MyModel()
    # make config file if it doesn't exist, otherwise load it and apply to model
    load_apply_save_config(model, filename)

or to iterate and test the model using a range of values for a property:

.. code-block:: python

    from tree_config import read_config_from_file, read_config_from_object, \
        apply_config

    model = MyModel()
    config = read_config_from_file(filename)
    # or
    config = read_config_from_object(model)

    for model_size in range(3, 6):
        config['model_size'] = model_size
        apply_config(model, config)
        train_model(...)

If any of the configurable properties are PyTorch tensors or numpy arrays,
one must first register their support before any configuration, as follows:

.. code-block:: python

    from tree_config.yaml import register_torch_yaml_support, \
    register_numpy_yaml_support

    register_numpy_yaml_support()
    register_torch_yaml_support()

.. _log-guide:

Logging
-------

Sapicore supports logging of arbitrary model (neuron, synapse, etc.) properties, including
scalars, tensors, and numpy arrays with an API similar to :ref:`config-guide`.

Each of the model, neuron, synapse etc classes inherit from
:py:class:`sapicore.logging.Loggable`. This adds support for using ``_loggable_props_`` to list
the names of all the properties of the class that is **potentially** logged.
Similarly, starting from the root model, ``_loggable_children_`` tells the logging
system about children objects of the model, recursively, that also support
logging of their properties.

Then we use :py:mod:`sapicore.logging` functions to get the dict of all properties
across all objects that **could** be logged, each mapped to ``True`` or ``False``
indicating whether it should actually be logged. This can be edited by the user
either when it's in the dict from or in a yaml file, to selectively enable the
properties to log.

Then to actually log the selected properties, one calls the log function (see below)
each time they are to be logged (e.g. every 5th iteration) and then the selected
properties are logged to the log file.

For example, we can define this model such that it contains an activation value that
is updated at every iteration:

.. code-block:: python

    class MyModel(SapicoreModel):

        _loggable_props_ = ('activation', )

        activation = 0

Then we can use :py:func:`~sapicore.logging.read_loggable_from_object`,
:py:func:`~sapicore.logging.read_loggable_from_file`, or
:py:func:`~sapicore.logging.update_loggable_from_object` to get the dict of properties to
be logged,
potentially further editing it turning ON or OFF some properties, and then use
:py:func:`~sapicore.logging.get_loggable_properties` to get a filtered list
of loggable properties ready to be directly used by the logging system. To dump the
loggable properties dict to a yaml file one uses :py:func:`~sapicore.logging.dump_loggable`.
Or we can just use :py:func:`~tree_config.load_save_get_loggable_properties` to do it all
in one step.

So, we could do:

.. code-block:: python

    sapicore.logging import load_save_get_loggable_properties, \
    read_loggable_from_object, get_loggable_properties

    model = MyModel()
    # make loggable file if it doesn't exist (logging all by default), otherwise load it
    props = load_save_get_loggable_properties(model, filename, default_value=True)

    # or instead edit the properties to be logged first
    log_opts = read_loggable_from_object(model, default_value=False)
    log_opts['activation'] = True
    props = get_loggable_properties(model, log_opts)

Once we have the filtered list of properties to be logged, Sapicore supports logging
them either to ``tensorboard`` using :py:func:`~sapicore.logging.log_tensor_board`
or to a Nix HDF5 file using :py:class:`~sapicore.logging.NixLogWriter`.

The tensorboard logging system is meant for live display of scalar summary data
in small volumes (see the ``tensorboard`` project) through
:py:class:`torch.utils.tensorboard.SummaryWriter`.
The Nix file, however, supports logging arbitrary tensors and numpy arrays and
can be used for broader debug or model performance logging. The logged data can
then be accessed after an experiment using :py:class:`~sapicore.logging.NixLogReader`.

For example to log to tensorboard, starting from the properties list:

.. code-block:: python

    writer = SummaryWriter(log_dir='tensorboard')
    for i in range(10):
        train_model(...)
        log_tensor_board(writer, props, global_step=i, prefix='example')
    writer.close()

Similarly, to log to Nix:

.. code-block:: python

    writer = NixLogWriter(filename, config_data=...)
    writer.create_file()
    writer.create_block('example')

    for i in range(10):
        train_model(...)
        writer.log(props, i)
    writer.close_file()

Then, to get the data from the file:

.. code-block:: python

    with NixLogReader(filename) as reader:
        print(reader.get_experiment_names())
        print(reader.get_experiment_property_paths('example'))
        print(reader.get_experiment_property_data('example', ('activation', )))

See those classes for full API details.
