from .experiment import make_experiment
from .framework.compete import TSCompeteExperiment
from .hyper_ts import HyperTS, HyperModel, HyperTSEstimator

from ._version import __version__


def _init():
    import warnings
    from hypernets.utils import logging, isnotebook

    warnings.filterwarnings('ignore')
    if isnotebook():
        logging.set_level('warn')


_init()