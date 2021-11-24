from hypernets.pipeline.base import HyperTransformer
from hypernets.pipeline.transformers import SimpleImputer


class TimeSeriesTransformer:

    def __init__(self, time_series_col=None):
        self.time_series_col = time_series_col

    def transform(self, X, y=None, **kwargs):
        # TODO:
        return X.values

    def fit(self, X, y=None, **kwargs):
        # TODO:
        return self


class TimeSeriesHyperTransformer(HyperTransformer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, TimeSeriesTransformer, space, name, **kwargs)

