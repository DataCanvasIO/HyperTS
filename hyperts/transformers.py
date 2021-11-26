import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hypernets.pipeline.base import HyperTransformer


class TimeSeriesTransformer:

    def __init__(self, time_series_col=None):
        self.time_series_col = time_series_col

    def transform(self, X, y=None, **kwargs):
        # TODO:
        return X.values

    def fit(self, X, y=None, **kwargs):
        # TODO:
        return self

class LogXplus1Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super(LogXplus1Transformer, self).__init__()

    def fit(self, X, y=None):
        X = np.log(X + 1)
        return X

    def transform(self, X):
        X = np.log(X + 1)
        return X

    def inverse_transform(self, X):
        X = np.exp(X) - 1
        return X



class TimeSeriesHyperTransformer(HyperTransformer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, TimeSeriesTransformer, space, name, **kwargs)

class LogXplus1HyperTransformer(HyperTransformer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, LogXplus1Transformer, space, name, **kwargs)
