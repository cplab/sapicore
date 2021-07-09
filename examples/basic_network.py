from os.path import join
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from tree_config.yaml import register_torch_yaml_support, \
    register_numpy_yaml_support
from tree_config import load_apply_save_config
import nixio as nix

from sapicore.model import SapicoreModel
from sapicore.neuron.analog import AnalogNeuron
from sapicore.synapse import SapicoreSynapse
from sapicore.pipeline import PipelineBase
from sapicore.learning import SapicoreLearning
from sapicore.logging import load_save_get_loggable_properties, \
    log_tensor_board, log_nix, load_nix_log, create_nix_file, \
    create_nix_logging_block


class SimpleNeuron(AnalogNeuron):

    _config_props_ = ('clip_min', 'clip_max')

    _loggable_props_ = ('activation', )

    activation: torch.Tensor
    """The neuron activation."""

    clip_min = -2.

    clip_max = 2.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # register randomness as part of the state so it is saved with the model
        self.register_buffer("randomness", torch.zeros(0))
        self.activation = torch.zeros(0)

    def forward(self, data: torch.tensor) -> torch.tensor:
        self.activation = torch.clip(data, self.clip_min, self.clip_max)
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

    def initialize_state(self, model_size, **kwargs):
        super().initialize_state(**kwargs)
        self.weight = torch.normal(self.mean, self.std, size=(model_size, ))
        self.activation = torch.zeros(0)

    def forward(self, data: torch.tensor) -> torch.tensor:
        # add the noise to the incoming signal
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

    seed = 0

    num_iterations = 3

    cuda_device: torch.device = None

    model: MyModel = None

    nix_file: nix.File = None

    nix_counter_block = None

    nix_prop_block = None

    def parse_cmd(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default='.')
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num_iterations", type=int, default=3)
        args = parser.parse_args()

        self.root_path = args.root
        self.seed = args.seed
        self.num_iterations = args.num_iterations

    def init_torch(self):
        use_cuda = torch.cuda.is_available()
        self.cuda_device = torch.device("cuda:0" if use_cuda else "cpu")
        # Sets up Gpu use
        if use_cuda:
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.manual_seed(self.seed)

    def init_model(self):
        model = self.model = MyModel()

        config_filename = join(self.root_path, 'config.yaml')
        config = load_apply_save_config(model, config_filename)

        debug_logging_filename = join(self.root_path, 'debug_logging.yaml')
        debug_logging = load_save_get_loggable_properties(
            model, debug_logging_filename, default_value=True)

        h5_filename = join(self.root_path, 'debug_logging.h5')
        nix_file = self.nix_file = create_nix_file(
            h5_filename, config_data=config)
        self.nix_counter_block, self.nix_prop_block = create_nix_logging_block(
            nix_file, 'simple_example', debug_logging)

        model.initialize_state()
        model.initialize_learning_state()
        model.to(self.cuda_device)

    def run_iteration(self, i):
        model = self.model
        with torch.no_grad():
            # fake data
            data = torch.normal(0, 1, size=(model.model_size, ))
            model.forward(data)
            model.apply_learning()

            log_nix(self.nix_counter_block, self.nix_prop_block, i)

    def teardown(self):
        model_filename = join(self.root_path, 'model.torch')
        torch.save(self.model.state_dict(), model_filename)

        self.nix_file.close()

    def run(self) -> None:
        self.parse_cmd()
        self.init_torch()
        self.init_model()

        for i in range(self.num_iterations):
            self.run_iteration(i)

        self.teardown()


if __name__ == '__main__':
    pipeline = SimplePipeline()
    pipeline.run()
