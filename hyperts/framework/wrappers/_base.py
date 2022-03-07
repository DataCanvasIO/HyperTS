import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from hypernets.core.search_space import ModuleSpace

from hyperts.utils import consts
from hyperts.utils._base import get_tool_box
from hyperts.utils.transformers import (LogXplus1Transformer,
                                        IdentityTransformer,
                                        StandardTransformer,
                                        MinMaxTransformer,
                                        MaxAbsTransformer)


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

        # fitted
        self.transformers = None
        self.sc = None
        self.lg = None

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
    def classes_(self):
        return None

    def fit_transform(self, X):
        tb = get_tool_box(X)
        if self.is_log is not None:
            self.lg = self.logx.get(self.is_log, None)
        if self.is_scale is not None:
            self.sc = self.scaler.get(self.is_scale, None)

        pipelines = []
        if self.is_log is not None:
            pipelines.append((f'{self.is_log}', self.lg))
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
                transform_X = pd.DataFrame(transform_X, columns=cols)
            else:
                transform_X = tb.from_3d_array_to_nested_df(transform_X, columns=cols)

        return transform_X

    def inverse_transform(self, X):
        try:
            inverse_X = self.transformers.inverse_transform(X)
        except:
            inverse_X = self.transformers._inverse_transform(X)
        return inverse_X

    def update_dl_kwargs(self):
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