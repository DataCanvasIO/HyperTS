def _init():
    import os
    import warnings
    from hypernets.utils import logging, isnotebook

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')
    if isnotebook():
        logging.set_level('warn')

_init()

from .experiment import make_experiment
from .framework.compete import TSCompeteExperiment
from .hyper_ts import HyperTS, HyperModel, HyperTSEstimator

from ._version import __version__