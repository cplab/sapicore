from os.path import join
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from tree_config.yaml import register_torch_yaml_support, \
    register_numpy_yaml_support
from tree_config import load_apply_save_config

from sapicore.model import SapicoreModel
from sapicore.neuron.analog import AnalogNeuron
from sapicore.synapse import SapicoreSynapse
from sapicore.pipeline import PipelineBase
from sapicore.learning import SapicoreLearning
from sapicore.logging import load_save_get_loggable_properties, \
    log_tensor_board, NixLogWriter, NixLogReader, read_loggable_from_object, \
    get_loggable_properties


class SimpleNeuron(AnalogNeuron):

    _config_props_ = ('clip_min', 'clip_max')

    _loggable_props_ = ('activation', 'intensity')

    activation: torch.Tensor
    """The neuron activation."""

    clip_min = -2.

    clip_max = 2.

    intensity = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = torch.zeros(0)

    def forward(self, data: torch.tensor) -> torch.tensor:
        self.activation = torch.clip(data, self.clip_min, self.clip_max)
        self.intensity = torch.sum(self.activation)
        return self.activation


class SimpleSynapse(SapicoreSynapse):

    _config_props_ = ('std', 'mean')

    _loggable_props_ = ('activation', )

    activation: torch.Tensor
    """The value in the synapse."""

    weight: torch.Tensor
    """Added to the input."""

    std = 1.

    mean = 0.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # register weight as part of the state so it is saved with the model
        self.register_buffer("weight", torch.zeros(0))
        self.weight = torch.zeros(0)
        self.activation = torch.zeros(0)

    def initialize_state(self, model_size, **kwargs):
        super().initialize_state(**kwargs)
        self.weight = torch.normal(self.mean, self.std, size=(model_size, ))

    def forward(self, data: torch.tensor) -> torch.tensor:
        self.activation = data * self.weight
        return self.activation


class SimpleLearning(SapicoreLearning):

    _config_props_ = ('attenuation', )

    attenuation = 1

    def apply_learning(
            self, pre_neuron: SimpleNeuron, synapse: SimpleSynapse,
            post_neuron: SimpleNeuron, **kwargs):
        synapse.weight *= torch.abs_(pre_neuron.activation) * \
            torch.abs_(post_neuron.activation) * self.attenuation


class MyModel(SapicoreModel):

    _config_props_ = ('model_size', )

    _config_children_ = {
        'neuron 1': 'neuron_1', 'synapse': 'synapse', 'neuron 2': 'neuron_2',
        'learning': 'learning'
    }

    _loggable_props_ = ('activation_sum', )

    _loggable_children = _config_children_

    model_size = 3

    neuron_1: SimpleNeuron

    synapse: SimpleSynapse

    neuron_2: SimpleNeuron

    learning: SimpleLearning

    activation_sum = 0

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

    def initialize_state(self, **kwargs):
        # pass model size to the neurons/synapse so it can set the right size
        super().initialize_state(model_size=self.model_size, **kwargs)

    def initialize_learning_state(self) -> None:
        self.learning.initialize_state()

    def forward(self, data: torch.tensor) -> torch.tensor:
        data = self.neuron_1(data)
        data = self.synapse(data)
        data = self.neuron_2(data)

        self.activation_sum = torch.sum(data)
        return data

    def apply_learning(self, **kwargs) -> None:
        self.learning.apply_learning(self.neuron_1, self.synapse, self.neuron_2)


class SimplePipeline(PipelineBase):

    root_path = ''
    """Where we'll save the logs.
    """

    seed = 0
    """Used to init the random number generators.
    """

    num_iterations = 3
    """Number of training iterations.
    """

    cuda_device: torch.device = None
    """The cude device to use. Defaults to GPU if available, otherwise CPU.
    """

    model: MyModel = None
    """The network model.
    """

    nix_writer: NixLogWriter = None
    """Object used to log debug to nix file.
    """

    property_arrays = None
    """Property objects used to debug log selected properties to the nix file.
    """

    tensorboard_writer: SummaryWriter = None
    """The tensor board log writer.
    """

    tensor_loggable_properties = None
    """The properties to be logged by tensorboard"""

    def parse_cmd(self):
        """Read cmd args.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default='.')
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num_iterations", type=int, default=3)
        args = parser.parse_args()

        self.root_path = args.root
        self.seed = args.seed
        self.num_iterations = args.num_iterations

    def init_torch(self):
        """Init torch library.
        """
        use_cuda = torch.cuda.is_available()
        self.cuda_device = torch.device("cuda:0" if use_cuda else "cpu")
        # Sets up Gpu use
        if use_cuda:
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.manual_seed(self.seed)

    def init_model(self):
        """Create and init the model.
        """
        model = self.model = MyModel()

        # create if needed and then load the user config options for the model
        config_filename = join(self.root_path, 'config.yaml')
        config = load_apply_save_config(model, config_filename)

        # create a logging config file to dump debug data to a nix file. The
        # file lets you select which property to log
        debug_logging_filename = join(self.root_path, 'debug_logging.yaml')
        debug_logging = load_save_get_loggable_properties(
            model, debug_logging_filename, default_value=True)

        # create writer and file for logging the debug data
        h5_filename = join(self.root_path, 'debug_logging.h5')
        writer = self.nix_writer = NixLogWriter(h5_filename, config_data=config)
        writer.create_file()
        writer.create_block('simple_example')
        self.property_arrays = writer.get_property_arrays(
            'simple_example', debug_logging)

        # create a log config dict to log small data to tensorboard, and select
        # to log just one property. Tensorboard supports only brief logs
        log_opts = read_loggable_from_object(model, False)
        # only log intensity of neuron 1
        log_opts['neuron 1']['intensity'] = True
        self.tensor_loggable_properties = get_loggable_properties(
            model, log_opts)
        self.tensorboard_writer = SummaryWriter(
            log_dir=join(self.root_path, 'tensorboardx'))

        # now init the model
        model.initialize_state()
        model.initialize_learning_state()
        model.to(self.cuda_device)

    def run_iteration(self, i):
        model = self.model
        # these models don't use gradients
        with torch.no_grad():
            # fake data
            data = torch.normal(0, 1, size=(model.model_size, ))
            # pass it through the model
            model.forward(data)
            # apply model learning
            model.apply_learning()

            # log data for this iteration
            self.nix_writer.log(self.property_arrays, i)
            log_tensor_board(
                self.tensorboard_writer, self.tensor_loggable_properties,
                global_step=i, prefix='simple_example')

    def teardown(self):
        """Close pipeline.
        """
        # dump trained model state to a file.
        model_filename = join(self.root_path, 'model.torch')
        torch.save(self.model.state_dict(), model_filename)

        self.nix_writer.close_file()
        self.tensorboard_writer.close()

    def run(self) -> None:
        """Runs a full training session.
        """
        self.parse_cmd()
        self.init_torch()
        self.init_model()

        for i in range(self.num_iterations):
            self.run_iteration(i)

        self.teardown()


if __name__ == '__main__':
    # create and run the model
    pipeline = SimplePipeline()
    pipeline.run()

    # print logged data
    with NixLogReader(join(pipeline.root_path, 'debug_logging.h5')) as reader:
        print('Logged experiments: ', reader.get_experiment_names())
        print('Logged properties: ',
              reader.get_experiment_property_paths('simple_example'))

        print('Logged neuron 1 intensity: ',
              reader.get_experiment_property_data(
                  'simple_example', ('neuron_1', 'intensity')))
        print('Logged neuron 1 activation: ',
              reader.get_experiment_property_data(
                  'simple_example', ('neuron_1', 'activation')))
        print('Logged model activation_sum: ',
              reader.get_experiment_property_data(
                  'simple_example', ('activation_sum', )))
