# -*- coding:utf-8 -*-

import numpy as np
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def ptp(X):
    """
    Range of values (maximum - minimum) along an axis.
    """
    num_metafeatures = X.shape[1]
    domain = np.zeros((num_metafeatures, 2))
    for i in range(num_metafeatures):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])

    return domain


def normalization(metafeatures):
    """
    Normalized meta features.
    """
    domain = ptp(np.array(metafeatures))
    normalize = lambda X: (X - domain[:, 0]) / np.ptp(domain, axis=1)
    normalized_metafeatures = normalize(metafeatures)

    return normalized_metafeatures


def warm_start_sample(space_sample, trial_store):
    """Initialize the HyperSpace with promising meta-hyperparameter.

    Parameters
    ----------
    space_sample: HyperSpace class.
    trial_store: TrialStore class.
    """
    sample_signature = space_sample.signature
    len_vectors = len(space_sample.vectors)

    suggest_vectors = None
    for trial in trial_store.trials:
        if not trial.run and trial.signature == sample_signature and \
                len(trial.vectors) == len_vectors:
            suggest_vectors = trial.vectors

    if suggest_vectors is not None:
        space_sample.assign_by_vectors(suggest_vectors)
    else:
        logger.info('No matching mata sample was found to warm strat.')

    return space_sample