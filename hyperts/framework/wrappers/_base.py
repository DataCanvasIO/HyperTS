import os
import time
import numpy as np
import pandas as pd

from scipy.stats import binom
from scipy.special import erf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets

from hypernets.utils import logging
from hypernets.core.search_space import ModuleSpace

from hyperts.utils import consts
from hyperts.utils._base import get_tool_box
from hyperts.utils.transformers import (LogXplus1Transformer,
                                        IdentityTransformer,
                                        StandardTransformer,
                                        MinMaxTransformer,
                                        MaxAbsTransformer,
                                        OutliersTransformer)

logger = logging.get_logger(__name__)


class EstimatorWrapper:
    """Abstract base class for time series estimator wrapper.

    Notes
    -------
    X:  For classification and regeression tasks, X are the time series
        variable features. For forecast task, X is the timestamps and
        other covariables.
    """
    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError(
            'fit is a protected abstract method, it must be implemented.'
        )

    def predict(self, X, **kwargs):
        raise NotImplementedError(
            'predict is a protected abstract method, it must be implemented.'
        )

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError(
            'predict_proba is a protected abstract method, it must be implemented.'
        )


class WrapperMixin:
    """Mixin class for all transformers in estimator wrapper.

    """
    def __init__(self, fit_kwargs, **kwargs):
        if fit_kwargs.get('timestamp') is not None:
            self.timestamp = fit_kwargs.pop('timestamp')
        elif kwargs.get('timestamp') is not None:
            self.timestamp = kwargs.get('timestamp')
        else:
            self.timestamp = consts.TIMESTAMP

        if fit_kwargs.get('covariates') is not None:
            self.covariates = fit_kwargs.pop('timestamp')
        elif kwargs.get('covariates') is not None:
            self.covariates = kwargs.get('covariates')
        else:
            self.covariates = None

        self.freq = kwargs.pop('freq', None)

        if kwargs.get('drop_sample_rate') is not None:
            self.drop_sample_rate = kwargs.pop('drop_sample_rate')
        else:
            self.drop_sample_rate = 0.

        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.init_kwargs = kwargs if kwargs is not None else {}

        if kwargs.get('x_scale') is not None:
            self.is_scale = kwargs.pop('x_scale', None)
        elif kwargs.get('y_scale') is not None:
            self.is_scale = kwargs.pop('y_scale', None)
        else:
            self.is_scale = None
        if kwargs.get('x_log') is not None:
            self.is_log = kwargs.pop('x_log', None)
        elif kwargs.get('y_log') is not None:
            self.is_log = kwargs.pop('y_log', None)
        else:
            self.is_log = None
        if kwargs.get('outlier') is not None:
            self.is_outlier = kwargs.pop('outlier', None)
        else:
            self.is_outlier = None

        # fitted
        self.transformers = None
        self.sc = None
        self.lg = None
        self.ol = None

    @property
    def logx(self):
        return {
            'logx': LogXplus1Transformer()
        }

    @property
    def scaler(self):
        return {
            'z_scale': StandardTransformer(),
            'min_max': MinMaxTransformer(),
            'max_abs': MaxAbsTransformer()
        }

    @property
    def outlier(self):
        return {
            'fill': OutliersTransformer('fill', freq=self.freq),
            'clip': OutliersTransformer('clip'),
        }

    @property
    def classes_(self):
        return None

    def fit_transform(self, X):
        tb = get_tool_box(X)
        if self.is_log is not None:
            self.lg = self.logx.get(self.is_log, None)
        if self.is_scale is not None:
            self.sc = self.scaler.get(self.is_scale, None)
        if self.is_outlier is not None:
            self.ol = self.outlier.get(self.is_outlier, None)

        pipelines = []
        if self.is_log is not None:
            pipelines.append((f'{self.is_log}', self.lg))
        if self.is_outlier is not None:
            pipelines.append((f'{self.is_outlier}', self.ol))
        if self.is_scale is not None:
            pipelines.append((f'{self.is_scale}', self.sc))
        pipelines.append(('identity', IdentityTransformer()))
        self.transformers = Pipeline(pipelines)

        cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        if tb.is_nested_dataframe(X):
            X = tb.from_nested_df_to_3d_array(X)

        transform_X = self.transformers.fit_transform(X)

        if isinstance(transform_X, np.ndarray):
            if len(transform_X.shape) == 2:
                transform_X = pd.DataFrame(transform_X, columns=cols)
            else:
                transform_X = tb.from_3d_array_to_nested_df(transform_X, columns=cols)

        return transform_X

    def transform(self, X):
        tb = get_tool_box(X)
        cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        if tb.is_nested_dataframe(X):
            X = tb.from_nested_df_to_3d_array(X)

        try:
            transform_X = self.transformers.transform(X)
        except:
            transform_X = self.transformers._transform(X)

        if isinstance(transform_X, np.ndarray):
            if len(transform_X.shape) == 2:
                transform_X = tb.DataFrame(transform_X, columns=cols)
            else:
                transform_X = tb.from_3d_array_to_nested_df(transform_X, columns=cols)

        return transform_X

    def inverse_transform(self, X):
        try:
            inverse_X = self.transformers.inverse_transform(X)
        except:
            inverse_X = self.transformers._inverse_transform(X)
        return inverse_X

    def drop_hist_sample(self, X, y=None, **kwargs):
        tb = get_tool_box(X)
        data_len = tb.get_shape(X)[0]
        if kwargs.get('window') is not None:
            if kwargs['window'] + kwargs['forecast_length'] + 1 > \
               int(data_len*(1 - self.drop_sample_rate)) // 2:
                return X, y
        X = tb.select_1d_reverse(X, int(data_len*(1 - self.drop_sample_rate)))
        X = tb.reset_index(X)
        if y is not None:
            y = tb.select_1d_reverse(y, int(data_len*(1 - self.drop_sample_rate)))
            y = tb.reset_index(y)
        return X, y

    def detection_split_XTC(self, XTC):
        tb = get_tool_box(XTC)
        all_var_cols = tb.columns_tolist(XTC)
        if self.covariates is None:
            ex_var_cols = [self.timestamp]
        else:
            ex_var_cols = [self.timestamp] + self.covariates
        x_var_cols = tb.list_diff(all_var_cols, ex_var_cols)
        X = XTC[x_var_cols]
        TC = tb.drop(XTC, columns=x_var_cols)

        X = tb.reset_index(X)
        TC = tb.reset_index(TC)

        return TC, X

    def update_init_kwargs(self, **kwargs):
        if kwargs.get('y_scale') is not None:
            if kwargs.get('y_scale') == 'min_max':
                kwargs['out_activation'] = 'sigmoid'
            elif kwargs.get('y_scale') == 'max_abs':
                kwargs['out_activation'] = 'tanh'
            else:
                kwargs['out_activation'] = 'linear'
        if kwargs.get('x_scale') is not None:
            if kwargs.get('x_scale') == 'min_max':
                kwargs['out_activation'] = 'sigmoid'
            else:
                kwargs['out_activation'] = 'linear'
        return kwargs

    def update_fit_kwargs(self):
        if self.init_kwargs.get('batch_size'):
            self.fit_kwargs.update({'batch_size': self.init_kwargs.pop('batch_size')})
        if self.init_kwargs.get('epochs'):
            self.fit_kwargs.update({'epochs': self.init_kwargs.pop('epochs')})
        if self.init_kwargs.get('verbose'):
            self.fit_kwargs.update({'verbose': self.init_kwargs.pop('verbose')})
        if self.init_kwargs.get('callbacks'):
            self.fit_kwargs.update({'callbacks': self.init_kwargs.pop('callbacks')})
        if self.init_kwargs.get('validation_split'):
            self.fit_kwargs.update({'validation_split': self.init_kwargs.pop('validation_split')})
        if self.init_kwargs.get('validation_data'):
            self.fit_kwargs.update({'validation_data': self.init_kwargs.pop('validation_data')})
        if self.init_kwargs.get('shuffle'):
            self.fit_kwargs.update({'shuffle': self.init_kwargs.pop('shuffle')})
        if self.init_kwargs.get('class_weight'):
            self.fit_kwargs.update({'class_weight': self.init_kwargs.pop('class_weight')})
        if self.init_kwargs.get('sample_weight'):
            self.fit_kwargs.update({'sample_weight': self.init_kwargs.pop('sample_weight')})
        if self.init_kwargs.get('initial_epoch'):
            self.fit_kwargs.update({'initial_epoch': self.init_kwargs.pop('initial_epoch')})
        if self.init_kwargs.get('steps_per_epoch'):
            self.fit_kwargs.update({'steps_per_epoch': self.init_kwargs.pop('steps_per_epoch')})
        if self.init_kwargs.get('validation_steps'):
            self.fit_kwargs.update({'validation_steps': self.init_kwargs.pop('validation_steps')})
        if self.init_kwargs.get('validation_freq'):
            self.fit_kwargs.update({'validation_freq': self.init_kwargs.pop('validation_freq')})
        if self.init_kwargs.get('max_queue_size'):
            self.fit_kwargs.update({'max_queue_size': self.init_kwargs.pop('max_queue_size')})
        if self.init_kwargs.get('workers'):
            self.fit_kwargs.update({'workers': self.init_kwargs.pop('workers')})
        if self.init_kwargs.get('use_multiprocessing'):
            self.fit_kwargs.update({'use_multiprocessing': self.init_kwargs.pop('use_multiprocessing')})

    def _merge_dict(self, *args):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
        return d


##################################### Define Simple Time Series Estimator #####################################
class SimpleTSEstimator(ModuleSpace):
    """A Simple Time Series Estimator.

    """
    def __init__(self, wrapper_cls, fit_kwargs=None, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.wrapper_cls = wrapper_cls
        self.estimator = None

    def build_estimator(self, task=None):
        pv = self.param_values
        self.estimator = self.wrapper_cls(self.fit_kwargs, **pv)
        return self.estimator

    def _forward(self, inputs):
        return self.estimator

    def _compile(self):
        pass


######################################## Define Base Anomaly Detector ########################################
class BaseAnomalyDetectorWrapper:
    """Abstract class for all anomaly detector.

    Parameters
    ----------
    name: str, the name of detection algorithm.

    contamination: float, the range in (0., 0.5), optional (default=0.05).
        The amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,).
        The outlier scores of the training data.

    threshold_ : float, the threshold is based on `contamination`.
        It is the `n_samples * contamination` most abnormal samples in
        `decision_scores_`. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1.
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        `threshold_` on `decision_scores_`.

    classes_: int, default 2.
        Default as binary classification.
    """

    def __init__(self, name, contamination=0.05):
        self.name = name
        self.contamination = contamination

        self.classes_ = 2
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None, **kwargs):
        """Fit time series model to training data.

        Parameters
        ----------
        X : numpy array og shape (n_samples, n_features).

        y : ignored in unsupervised methods. default None.

        Returns
        -------
        self : object.
        """
        start = time.time()
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self._set_num_classes(y)

        self._fit(X=X, y=y, **kwargs)

        logger.info(f'Training finished, total taken {time.time() - start}s.')

        return self

    def predict(self, X, **kwargs):
        """Predict labels for sequences in X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features).

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """
        start = time.time()
        self._check_is_fitted()

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        pred = self._predict(X=X, **kwargs)

        logger.info(f'Training finished, total taken {time.time() - start}s.')

        return pred

    def predict_proba(self, X, methed='erf'):
        """Predict the probability for sequences in X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features).
        methed : str, optional {'erf', 'linear'}. Probability conversion method.

        Returns
        -------
        outlier_probability : numpy array of shape (n_samples, n_classes)
            For each observation, tells whether or not it should be considered
            as an outlier according to the fitted model. Return the outlier
            probability, ranging in [0,1]. Note it depends on the number of
            classes, which is by default 2 classes ([proba of normal, proba of outliers]).
        """
        self._check_is_fitted()

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        train_scores = self.decision_scores_
        mu = np.mean(train_scores)
        sigma = np.std(train_scores)

        test_scores = self.decision_function(X)

        probas = np.zeros((X.shape[0], self.classes_))

        if methed == 'linear':
            scaler = MinMaxScaler((0, 1))
            scaler.fit(train_scores.reshape(-1, 1))
            pr = scaler.transform(test_scores.reshape(-1, 1))
        else:
            pre_erf_score = (test_scores - mu) / (sigma * np.sqrt(24))
            pr = erf(pre_erf_score)

        pr = pr.ravel().clip(0, 1)
        probas[:, 0] = 1. - pr
        probas[:, 1] = pr

        return probas

    def predict_confidence(self, X):
        """Predict the confidence of model in making the same prediction
           under slightly different training sets.

        Parameters
        -------
        X : numpy array of shape (n_samples, n_features).

        Returns
        -------
        confidence : numpy array of shape (n_samples,).
            For each observation, tells how consistently the model would
            make the same prediction if the training set was perturbed.
            Return a probability, ranging in [0,1].

        Reference
        ---------
        https://github.com/yzhao062/pyod
        """
        self._check_is_fitted()

        if isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        test_scores = self.decision_function(X)

        nb_train_samples = len(self.decision_scores_)

        count_instances = np.vectorize(lambda x: np.count_nonzero(self.decision_scores_ <= x))
        nb_test_instances =  count_instances(test_scores)

        posterior_prob = np.vectorize(lambda x: (1+x)/(2+nb_train_samples))(nb_test_instances)

        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(nb_train_samples - int(nb_train_samples * self.contamination),
            nb_train_samples, p))(posterior_prob)

        prediction = (test_scores > self.threshold_).astype('int').ravel()
        np.place(confidence, prediction == 0, 1 - confidence[prediction == 0])

        return confidence

    def _fit(self, X, y=None, **kwargs):
        """Fit time series model to training data.

        """
        raise NotImplementedError(
            '_fit is a protected abstract method, it must be implemented.'
        )

    def _predict(self, X, **kwargs):
        """Predict labels for sequences in X.

        """
        raise NotImplementedError(
            '_predict is a protected abstract method, it must be implemented.'
        )

    def decision_function(self, X):
        """Predict anomaly scores for sequences in X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features).

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        raise NotImplementedError('Not be implemented.')

    def _get_decision_attributes(self):
        """Calculate key attributes: threshold_ and labels_.

        Returns
        -------
        self : object.
        """

        self.threshold_ = np.percentile(self.decision_scores_, 100*(1-self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        return self

    def _check_is_fitted(self):
        """Check if key attributes 'decision_scores_', 'threshold_',
           and 'labels_' are None.

        Returns
        -------
        True or False.
        """
        if self.decision_scores_ is None:
            return False
        elif self.threshold_ is None:
            return False
        elif self.labels_ is None:
            return False
        else:
            return True

    def _set_num_classes(self, y):
        """Set the number of classes if y is not None.

        Returns
        -------
        self : object.
        """
        if y is not None:
            check_classification_targets(y)
            self.classes_ = len(np.unique(y))
        return self


######################################## Other Support Component Library ########################################
class suppress_stdout_stderr:
    ''' Suppressing Stan optimizer printing in Prophet Wrapper.
        A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    References
    ----------
    https://github.com/facebook/prophet/issues/223
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)