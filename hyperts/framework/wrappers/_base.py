import os
import numpy as np
import pandas as pd

from hyperts.utils import consts
from hyperts.utils.transformers import LogXplus1Transformer, IdentityTransformer

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline


class EstimatorWrapper:

    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        """
        X:  For classification and regeression tasks, X are the time series
            variable features. For forecast task, X is the timestamps and
            other covariables.
        """
        raise NotImplementedError

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError


class WrapperMixin:

    def __init__(self, fit_kwargs, **kwargs):
        self.trans = None
        self.log = None
        self.scale = None
        self.sc = None
        self.lg = None

        self.timestamp = fit_kwargs.get('timestamp', consts.TIMESTAMP)
        self.init_kwargs = kwargs if kwargs is not None else {}

        if kwargs.get('x_scale') is not None:
            self.scale = kwargs.pop('x_scale', None)
        elif kwargs.get('y_scale') is not None:
            self.scale = kwargs.pop('y_scale', None)
        if kwargs.get('x_log') is not None:
            self.log = kwargs.pop('x_log', None)
        elif kwargs.get('y_log') is not None:
            self.log = kwargs.pop('y_log', None)

    @property
    def logx(self):
        return {
            'logx': LogXplus1Transformer()
        }

    @property
    def scaler(self):
        return {
            'min_max': MinMaxScaler(),
            'max_abs': MaxAbsScaler()
        }

    def fit_transform(self, X):
        if self.log is not None:
            self.lg = self.logx.get(self.log, None)
        if self.scale is not None:
            self.sc = self.scaler.get(self.scale, None)

        pipelines = []
        if self.log is not None:
            pipelines.append((f'{self.log}', self.lg))
        if self.sc is not None:
            pipelines.append((f'{self.scale}', self.sc))
        pipelines.append(('identity', IdentityTransformer()))
        self.trans = Pipeline(pipelines)

        cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        transform_X = self.trans.fit_transform(X)
        if isinstance(transform_X, np.ndarray):
            transform_X = pd.DataFrame(transform_X, columns=cols)

        return transform_X

    def inverse_transform(self, X):
        inverse_X = self.trans._inverse_transform(X)
        return inverse_X


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