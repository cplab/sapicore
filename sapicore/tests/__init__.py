""" Unit, integration, and functional tests. """
import os
import torch

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
""" Project root directory (.../sapicore), for use by test suite scripts. """

TEST_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
""" Hardware device to use during testing. Defaults to GPU when available, CPU otherwise. """
