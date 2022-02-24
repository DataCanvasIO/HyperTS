import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from hyperts.utils import consts
from hyperts.utils._base import get_tool_box
from hypernets.pipeline.base import HyperTransformer


##################################### Define sklearn Transformer #####################################
class TimeSeriesTransformer:
    """Scale time series features.

    """
    def __init__(self, time_series_col=None):
        self.time_series_col = time_series_col

    def transform(self, X, y=None, **kwargs):
        # TODO:
        return X.values

    def fit(self, X, y=None, **kwargs):
        # TODO:
        return self


class LogXplus1Transformer(BaseEstimator, TransformerMixin):
    """Scale each feature by log(x+1).

    """
    def __init__(self):
        super(LogXplus1Transformer, self).__init__()

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        transform_X = np.log(X + 1)
        transform_X = np.clip(transform_X, 1e-6, abs(transform_X))
        return transform_X

    def inverse_transform(self, X, y=None, **kwargs):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = np.exp(X) - 1
        return X


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer.

    """
    def __init__(self):
        super(IdentityTransformer, self).__init__()

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return X

    def inverse_transform(self, X, y=None, **kwargs):
        return X


class StandardTransformer(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance.

    Notes
    ----------
    Unlike scikit-learn, it can process 3D time series - (nb_samples, series_length, nb_dims).

    The transformation is given by::

        X_scaled = (X - X.mean) / (X.var + eps),

    where, for 2D features:
        mean = X.mean(axis=0),
        var  = ((X - mean) ** 2).mean(axis=0)
    for 3D features:
        mean = X.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True),
        var  = ((X - mean) ** 2).mean(axis=0, keepdims=True).mean(axis=1, keepdims=True).

    Parameters
    ----------
    eps  : float, default=1e-8.
        To prevent the division by 0.
    copy : bool, default=True.
        Set to False to perform inplace row normalization and avoid a copy.
    """
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
    """Transform features by scaling each feature to a given range.

    Notes
    ----------
    Unlike scikit-learn, it can process 3D time series - (nb_samples, series_length, nb_dims).

    The transformation is given by::

        X_scaled = (X - X.min) / (X.max - X.min + eps),

    where, for 2D features:
        min = X.min(axis=0, initial=None),
        max = X.max(axis=0, initial=None),
    for 3D features:
        min = X.min(axis=0, keepdims=True, initial=None).min(axis=1, keepdims=True),
        max = X.max(axis=0, keepdims=True, initial=None).max(axis=1, keepdims=True).

    Parameters
    ----------
    eps  : float, default=1e-8.
        To prevent the division by 0.
    copy : bool, default=True.
        Set to False to perform inplace row normalization and avoid a copy.
    """
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
    """Scale each feature by its maximum absolute value.

    Notes
    ----------
    Unlike scikit-learn, it can process 3D time series - (nb_samples, series_length, nb_dims).

    The transformation is given by::

        X_scaled = X / (X.max_abs + eps),

    where, for 2D features:
        max_abs = np.max(np.abs(X), axis=0),
    for 3D features:
        max_abs = np.abs(X).max(axis=0, keepdims=True).max(axis=1, keepdims=True)

    Parameters
    ----------
    eps  : float, default=1e-8.
        To prevent the division by 0.
    copy : bool, default=True.
        Set to False to perform inplace row normalization and avoid a copy.
    """
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
            self.max_abs = np.max(np.abs(X), axis=0)
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
    """Transform categorical labels to one hot labels.

    Parameters
    ----------
    label_encoder : An existing Label encoder, default=None.
    onehot_encoder : An existing OneHot encoder, default=None.
    copy : bool, default=True.
        Set to False to perform inplace row normalization and avoid a copy.
    """
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


class CovariateTransformer(BaseEstimator, TransformerMixin):
    """Transform covariates by 'drop_constant_columns', 'drop_duplicated_columns',
        'drop_idness_columns', 'replace_inf_values' and so on.

    Parameters
    ----------
    covariables: list[n*str], if the data contains covariables, specify the
        covariable column names, (default=None).
    data_cleaner_args : dict or None, (default=None).
        If not None, the definition example is as follows:
            data_cleaner_args = {
                'correct_object_dtype': False,
                'int_convert_to': 'str',
                'drop_constant_columns': True,
                'drop_duplicated_columns': True,
                'drop_idness_columns': True,
                'replace_inf_values': np.nan,
                ...
            }
    Reference for details: https://github.com/DataCanvasIO/Hypernets/blob/master/hypernets/tabular/data_cleaner.py
    """
    def __init__(self, covariables, data_cleaner_args=None):
        super(CovariateTransformer, self).__init__()
        self.covariables = covariables
        if data_cleaner_args is None:
            self.data_cleaner_args = {'correct_object_dtype': False,
                                      'int_convert_to': 'str'}
        else:
            self.data_cleaner_args = data_cleaner_args

        self.cleaner = None
        self.covariables_ = None
        self.dorp_nan_columns = []

    def fit(self, X, y=None, **kwargs):
        tb = get_tool_box(X)
        null_num = X[self.covariables].isnull().sum().to_dict()
        for k, v in null_num.items():
            if v > len(X)*consts.NAN_DROP_SIZE:
                self.dorp_nan_columns.append(k)
        X = X.drop(columns=self.dorp_nan_columns)
        self.covariables = tb.list_diff(self.covariables, self.dorp_nan_columns)
        self.cleaner = tb.data_cleaner(**self.data_cleaner_args)
        covariates, _ = self.cleaner.fit_transform(X[self.covariables])
        self.covariables_ = covariates.columns.to_list()
        return self

    def transform(self, X, y=None, **kwargs):
        tb = get_tool_box(X)
        X = X.drop(columns=self.dorp_nan_columns)
        covariates = self.cleaner.transform(X[self.covariables])
        X = X.drop(columns=self.covariables_ if self.dorp_nan_columns else self.covariables)
        X = tb.concat_df([X, covariates], axis=1)
        return X


##################################### Define Hyper Transformer #####################################
class TimeSeriesHyperTransformer(HyperTransformer):

    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, TimeSeriesTransformer, space, name, **kwargs)

class LogXplus1HyperTransformer(HyperTransformer):

    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, LogXplus1Transformer, space, name, **kwargs)
