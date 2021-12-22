import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from hypernets.pipeline.base import HyperTransformer


##################################### Define sklearn Transformer #####################################
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

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.log(X + 1)
        return X

    def inverse_transform(self, X, y=None, **kwargs):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.exp(X) - 1
        return X


class IdentityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super(IdentityTransformer, self).__init__()

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return X

    def inverse_transform(self, X, y=None, **kwargs):
        return X


class StandardTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, eps=1e-8, copy=True):
        super(StandardTransformer, self).__init__()
        self.eps = eps
        self.copy = copy
        self.mean = None
        self.var = None

    def fit(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) == 2:
            self.mean = X.mean(axis=0)
            self.var = ((X - self.mean) ** 2).mean(axis=0)
        else:
            self.mean = X.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
            self.var = ((X - self.mean) ** 2).mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)

        return self

    def transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        transform_X = (X - self.mean) / np.sqrt(self.var + self.eps)
        return transform_X

    def inverse_transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        inverse_X = X * np.sqrt(self.var + self.eps) + self.mean
        return inverse_X


class MinMaxTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, eps=1e-8, copy=True):
        super(MinMaxTransformer, self).__init__()
        self.eps = eps
        self.copy = copy
        self.min = None
        self.max = None

    def fit(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) == 2:
            self.min = X.min(axis=0, initial=None)
            self.max = X.max(axis=0, initial=None)
        else:
            self.min = X.min(axis=0, keepdims=True, initial=None).min(axis=1, keepdims=True)
            self.max = X.max(axis=0, keepdims=True, initial=None).max(axis=1, keepdims=True)

        return self

    def transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        transform_X = (X - self.min) / (self.max - self.min + self.eps)
        return transform_X

    def inverse_transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        inverse_X = X * (self.max - self.min + self.eps) + self.min
        return inverse_X


class MaxAbsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, eps=1e-8, copy=True):
        super(MaxAbsTransformer, self).__init__()
        self.eps = eps
        self.copy = copy
        self.max_abs = None

    def fit(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) == 2:
            self.max_abs = np.nanmax(np.abs(X), axis=0)
        else:
            X = np.abs(X)
            self.max_abs = X.max(axis=0, keepdims=True).max(axis=1, keepdims=True)

        return self

    def transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        transform_X = X / (self.max_abs + self.eps)
        return transform_X

    def inverse_transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        inverse_X = X * (self.max_abs + self.eps)
        return inverse_X


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, label_encoder=None, onehot_encoder=None, copy=True):
        super(CategoricalTransformer, self).__init__()
        self.copy = copy
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
        else:
            self.label_encoder = label_encoder
        if onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        else:
            self.onehot_encoder = onehot_encoder

        self.classes_ = None
        self.nb_classes_ = None

    def fit(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        self.label_encoder.fit(X)
        self.classes_ = self.label_encoder.classes_
        self.nb_classes_ = len(self.classes_)
        X = self.label_encoder.transform(X)
        self.onehot_encoder.fit(X.reshape(len(X), 1))
        return self

    def transform(self, X, y=None,**kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        transform_X = self.label_encoder.transform(X)
        if self.nb_classes_ > 2: # multiclass
            transform_X = self.onehot_encoder.transform(transform_X.reshape(len(X), 1))
        return transform_X

    def inverse_transform(self, X, y=None, **kwargs):
        if self.copy:
            X = copy.deepcopy(X)
        if self.nb_classes_ > 2: # multiclass
            X = self.onehot_encoder.inverse_transform(X)
        inverse_X = self.label_encoder.inverse_transform(X)
        return inverse_X

##################################### Define Hyper Transformer #####################################
class TimeSeriesHyperTransformer(HyperTransformer):

    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, TimeSeriesTransformer, space, name, **kwargs)

class LogXplus1HyperTransformer(HyperTransformer):

    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, LogXplus1Transformer, space, name, **kwargs)
