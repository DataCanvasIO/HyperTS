import numpy as np
from sklearn.metrics import *


def mse(y_true, y_pred):
    """Mean squared error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    return np.nanmean((y_true - y_pred)**2, axis=0)


def mae(y_true, y_pred):
    """Mean absolute error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    return np.nanmean(np.abs(y_pred - y_true), axis=0)


def rmse(y_true, y_pred):
    """Root mean squared error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred, epsihon=1e-06, mask=False):
    """Mean absolute percentage error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    epsihon: float, threshold to avoid division by zero. Default is 1e-06.

    mask: bool, if True, the mask removes y_ture=0. Default is False.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    masks = y_true!=0. if mask else y_true==y_true
    diff = np.abs((y_pred[masks] - y_true[masks]) / np.clip(np.abs(y_true[masks]), epsihon, None))
    return np.nanmean(diff, axis=0)


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    diff = np.nanmean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)), axis=0)
    return 2.0 * diff
