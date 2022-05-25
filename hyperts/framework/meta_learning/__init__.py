import numpy as np

from .tsfeatures import metafeatures_from_timeseries


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