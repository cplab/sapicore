""" Train a model on a real-world dataset with cross validation. """
import os
import logging
from argparse import ArgumentParser

import torch
from torch import Tensor

from torch.nn.functional import normalize
from sklearn.model_selection import StratifiedKFold

from sapicore.data.sampling import BalancedSampler, CV
from sapicore.data.external.drift import DriftDataset

from sapicore.model import Model
from sapicore.pipeline import Pipeline
from sapicore.engine.network import Network

from sapicore.utils.io import log_settings
from sapicore.utils.seed import fix_random_seed
from sapicore.tests import ROOT

GAIN = 200.0
TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")


class EPL(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, data: Tensor) -> Tensor:
        """Predicts the labels of `data` by feeding the samples to a trained network and applying
        some procedure to the resulting population/readout layer response.

        Note
        ----
        SNNs trained with STDP produce differentiated population responses to different inputs. :meth:`predict` takes
        these responses and, by some algorithm (e.g., kNN), converts them to class labels using a custom procedure.

        An instructive version of this method will be implemented in a future tutorial.

        """
        raise NotImplementedError


class DriftExperiment(Pipeline):
    """Instantiates a minimal EPL network using the dictionary API and trains it on UCSD drift data."""

    def __init__(
        self,
        configuration: str | dict,
        stim_duration: int = 500,
        cv_folds: int = 2,
        log_dir: str = None,
        device: str = None,
        **kwargs,
    ):

        super().__init__(configuration=configuration, cv_folds=cv_folds, **kwargs)

        self.device = device
        self.log_dir = log_dir
        self.cv_folds = cv_folds
        self.stim_duration = stim_duration

        # apply logging configuration.
        log_settings()

        # for reproducibility.
        fix_random_seed(9846)

    def run(self):
        logging.info("Loading UCSD drift dataset.")
        drift = DriftDataset(root=os.path.join(TEST_ROOT, "drift"))()

        # since the full drift set has 13910 entries, sample 3 of each batch X chemical combination (171).
        # alternatively, sampling can be stratified with `n` representing a fraction of the total dataset.
        drift_subset = drift.sample(
            method=BalancedSampler(replace=False, stratified=False),
            group_keys=["batch", "chemical"],
            n=3,
        )

        # take only the first feature (`DR`) from each sensor (there are 8 features X 16 sensors).
        total_features = drift_subset.samples.shape[1]
        drift_subset = drift_subset.sample(lambda: torch.arange(0, total_features, 8), axis=1)

        # move data tensor to GPU if necessary.
        drift_subset.samples = drift_subset.samples.to(self.device)

        # L1-normalize features and multiply by desired gain factor.
        drift_subset.samples = GAIN * normalize(drift_subset.samples, p=1.0)

        # set up a cross validation object for our experiment.
        cv = CV(data=drift_subset, cross_validator=StratifiedKFold(self.cv_folds, shuffle=True), label_keys="chemical")

        for i, (train, test) in enumerate(cv):
            logging.info("Instantiating a new copy of the EPL network.")
            model = EPL(network=Network(configuration=self.configuration, device=self.device))

            logging.info(
                f"CV fold {i+1} with {len(train)} samples, "
                f"each sustained for {self.stim_duration} simulation steps (ms)."
            )

            # maintains each sample `stim_duration` steps (milliseconds).
            model.fit(drift_subset[train], repetitions=self.stim_duration)

            # save trained network objects for future use if a path was provided as a runtime argument.
            if self.log_dir:
                logging.info(f"Saving trained network to {self.log_dir}.")
                model.save(os.path.join(self.log_dir, f"EPL-Drift-Fold_{i}.pt"))

                # demonstrate reloading a trained network (overwriting the existing model.network attribute).
                model.load(os.path.join(self.log_dir, f"EPL-Drift-Fold_{i}.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-config",
        action="store",
        dest="config",
        metavar="FILE",
        required=True,
        help="Path to a model-simulation configuration YAML.",
    )
    parser.add_argument(
        "-out",
        action="store",
        dest="output",
        metavar="FILE",
        required=False,
        help="Destination path for storing trained model .pt files.",
    )
    args = parser.parse_args()

    # run the experiment from the given configuration dictionary.
    DriftExperiment(
        configuration=args.config,
        log_dir=args.output,
        stim_duration=25,
        cv_folds=2,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ).run()
