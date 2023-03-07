""" Train a model on a real-world dataset with cross validation. """
import os
import logging
from argparse import ArgumentParser

from torch import Tensor
from sklearn.model_selection import StratifiedKFold

from sapicore.data import Data
from sapicore.data.sampling import CV
from sapicore.data.external.drift import DriftDataset

from sapicore.pipeline import Pipeline
from sapicore.engine.network import Network
from sapicore.model import Model

from sapicore.utils.io import log_settings
from sapicore.utils.seed import fix_random_seed
from sapicore.tests import ROOT

TEST_ROOT = os.path.join(ROOT, "tests", "data", "test_data")


class EPL(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, data: Tensor) -> Tensor:
        """Predicts the labels of `data` by some procedure.

        Warning
        -------
        The implementation will be specific to this network and needs to be discussed.

        """
        pass


class DriftExperiment(Pipeline):
    """Instantiates a minimal EPL network using the dictionary API and trains it on UCSD drift data."""

    def __init__(self, configuration: str | dict, stim_duration: int = 500, cv_folds: int = 2, **kwargs):
        super().__init__(configuration=configuration, cv_folds=cv_folds, **kwargs)

        self.cv_folds = cv_folds
        self.stim_duration = stim_duration

        # apply logging configuration.
        log_settings()

        # for reproducibility.
        fix_random_seed(9846)

    def run(self):
        # since the full drift set has 13910 entries, randomly sample 130 of them in a balanced fashion.
        logging.info("Loading UCSD drift dataset.")
        drift = self.trim_stratified(retain=130)

        # fresh EPL model for each cross validation fold, each instantiated with a unique EPL network object.
        logging.info(f"Building {self.cv_folds} copies of the EPL network (128 MCs, 512 GCs, all-to-all).")
        models = [EPL(network=Network(configuration=self.configuration)) for _ in range(self.cv_folds)]

        # set up a cross validation object for our experiment.
        cv = CV(data=drift, cross_validator=StratifiedKFold(self.cv_folds, shuffle=True), label_key="chemical")

        # let the experiment commence.
        for i, (train, test) in enumerate(cv):
            logging.info(
                f"Simulating CV fold {i+1} with {len(train)} samples, " f"each sustained for {self.stim_duration}ms."
            )

            # maintains each sample `stim_duration` steps (milliseconds).
            models[i].fit(drift[train], repetitions=self.stim_duration)

    @staticmethod
    def trim_stratified(retain: int | float) -> Data:
        """Randomly sample from the drift dataset in a balanced fashion."""
        # fetch/load UCSD drift dataset.
        drift = DriftDataset(
            root=os.path.join(TEST_ROOT, "drift"),
            remote_urls="https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip",
        )()
        labels = drift.descriptors["chemical"].labels

        # leverage StratifiedKFold for balanced random sampling, retaining `retain` samples in the returned set.
        folds = int(drift.samples.shape[0] * retain) if isinstance(retain, float) else drift.samples.shape[0] // retain
        cv = CV(data=drift, cross_validator=StratifiedKFold(folds, shuffle=True), label_key="chemical")

        _, manageable = next(iter(cv))
        drift.samples = drift[manageable]
        drift.descriptors["chemical"].labels = labels[manageable]

        return drift


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
    args = parser.parse_args()

    # run the experiment from the given configuration file.
    DriftExperiment(configuration=args.config, stim_duration=100, cv_folds=2).run()
