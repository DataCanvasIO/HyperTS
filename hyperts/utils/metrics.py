import numpy as np
from sklearn.metrics import *
from hypernets.tabular import metrics
from hyperts.utils import consts as const

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def check_is_array(y_true, y_pred):
    """Check whether the value is array-like.
    If not, convert the value to array-like.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    return y_true, y_pred


def mse(y_true, y_pred, axis=None):
    """Mean squared error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    return np.nanmean((y_true - y_pred)**2, axis=axis)


def mae(y_true, y_pred, axis=None):
    """Mean absolute error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    return np.nanmean(np.abs(y_pred - y_true), axis=axis)


def rmse(y_true, y_pred):
    """Root mean squared error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred, epsihon=1e-06, mask=False, axis=None):
    """Mean absolute percentage error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    epsihon: float, threshold to avoid division by zero. Default is 1e-06.

    mask: bool, if True, the mask removes y_ture=0. Default is False.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    masks = y_true!=0. if mask else y_true==y_true
    diff = np.abs((y_pred[masks] - y_true[masks]) / np.clip(np.abs(y_true[masks]), epsihon, None))
    return np.nanmean(diff, axis=axis)


def smape(y_true, y_pred, axis=None):
    """Symmetric mean absolute percentage error.

    Note that this implementation can handle NaN.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    diff = np.nanmean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)), axis=axis)
    return 2.0 * diff

def msle(y_true, y_pred, epsihon=1e-06, axis=None):
    """Mean squared logarithmic error regression loss.

    Note that this implementation can handle NaN and y_pred contains negative values.

    Parameters
    ----------
    y_true : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : pd.DataFrame or array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_is_array(y_true, y_pred)

    if (y_true < 0).any():
        y_true = np.clip(y_true, a_min=epsihon, a_max=abs(y_true))

    if (y_pred < 0).any():
        y_pred = np.clip(y_pred, a_min=epsihon, a_max=abs(y_pred))

    return mse(np.log1p(y_true), np.log1p(y_pred), axis)


def auc(y_true, y_score, average="macro", sample_weight=None,
        max_fpr=None, multi_class="raise", labels=None):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see sklearn.metrics.roc_auc_score).

    """
    return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight,
                         max_fpr=max_fpr, multi_class=multi_class, labels=labels)


def _task_to_average(task):
    if 'binary' in task:
        average = 'binary'
    elif 'multiclass' in task:
        average = 'macro'
    else:
        average = None
    return average


def calc_score(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY,
               pos_label=1, classes=None, average=None):
    score = {}
    if y_proba is None:
        y_proba = y_preds

    if average is None:
        average = _task_to_average(task)

    recall_options = dict(average=average, labels=classes)

    if task in [const.TASK_BINARY, const.TASK_MULTICLASS] and pos_label is None:
        if 1 in y_true:
            recall_options['pos_label'] = 1
        elif 'yes' in y_true:
            recall_options['pos_label'] = 'yes'
        elif 'true' in y_true:
            recall_options['pos_label'] = 'true'
        else:
            recall_options['pos_label'] = y_true[0]
        logger.warning(f"pos_label is not specified and defaults to {recall_options['pos_label']}.")
    elif task in [const.TASK_BINARY, const.TASK_MULTICLASS] and pos_label is not None:
        if pos_label in y_true:
            recall_options['pos_label'] = pos_label
        else:
            recall_options['pos_label'] = y_true[0]
            logger.warning(f"pos_label is incorrect and defaults to {y_true[0]}.")
    else:
        recall_options['pos_label'] = None

    for metric in metrics:
        if callable(metric):
            if metric.__name__ in ['auc', 'roc_auc_score']:
                if len(y_proba.shape) == 2:
                    if 'multiclass' in task:
                        score[metric.__name__] = metric(y_true, y_proba, multi_class='ovo', labels=classes)
                    else:
                        score[metric.__name__] = metric(y_true, y_proba[:, 1])
                else:
                    score[metric.__name__] = metric(y_true, y_proba)
            else:
                try:
                    score[metric.__name__] = metric(y_true, y_preds)
                except:
                    score[metric.__name__] = metric(y_true, y_preds, **recall_options)
        else:
            metric_lower = metric.lower()
            if metric_lower in ['auc', 'roc_auc_score']:
                if len(y_proba.shape) == 2:
                    if 'multiclass' in task:
                        score[metric] = roc_auc_score(y_true, y_proba, multi_class='ovo', labels=classes)
                    else:
                        score[metric] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    score[metric] = roc_auc_score(y_true, y_proba)
            elif metric_lower == 'accuracy':
                if y_preds is None:
                    score[metric] = 0
                else:
                    score[metric] = accuracy_score(y_true, y_preds)
            elif metric_lower in ['recall']:
                score[metric] = recall_score(y_true, y_preds, **recall_options)
            elif metric_lower in ['precision']:
                score[metric] = precision_score(y_true, y_preds, **recall_options)
            elif metric_lower in ['f1']:
                score[metric] = f1_score(y_true, y_preds, **recall_options)
            elif metric_lower in ['mse', 'mean_squared_error', 'neg_mean_squared_error']:
                try:
                    score[metric] = mean_squared_error(y_true, y_preds)
                except:
                    score[metric] = mse(y_true, y_preds)
            elif metric_lower in ['mae', 'mean_absolute_error', 'neg_mean_absolute_error']:
                try:
                    score[metric] = mean_absolute_error(y_true, y_preds)
                except:
                    score[metric] = mae(y_true, y_preds)
            elif metric_lower in ['msle', 'mean_squared_log_error', 'neg_mean_squared_log_error']:
                try:
                    score[metric] = mean_squared_log_error(y_true, y_preds)
                except:
                    score[metric] = msle(y_true, y_preds)
            elif metric_lower in ['rmse', 'root_mean_squared_error', 'neg_root_mean_squared_error']:
                try:
                    score[metric] = mean_squared_error(y_true, y_preds, squared=False)
                except:
                    score[metric] = rmse(y_true, y_preds)
            elif metric_lower in ['mape', 'mean_absolute_percentage_error']:
                try:
                    score[metric] = mean_absolute_percentage_error(y_true, y_preds)
                except:
                    score[metric] = mape(y_true, y_preds)
            elif metric_lower in ['smape']:
                score[metric] = smape(y_true, y_preds)
            elif metric_lower in ['r2', 'r2_score']:
                score[metric] = r2_score(y_true, y_preds)
            elif metric_lower in ['logloss', 'log_loss']:
                score[metric] = log_loss(y_true, y_proba, labels=classes)
            else:
                logger.error(f'{metric_lower} is not supported. Therefore, reset reward_metric.')

    return score

metric2scoring = {
    'auc': 'roc_auc_ovo',
    'accuracy': 'accuracy',
    'accuracy_score': 'accuracy',
    'recall': 'recall',
    'recall_score': 'recall',
    'precision': 'precision',
    'precision_score': 'precision',
    'f1': 'f1',
    'f1_score': 'f1',
    'mse': 'neg_mean_squared_error',
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'mean_squared_error': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'mean_absolute_error': 'neg_mean_absolute_error',
    'neg_mean_squared_log_error': 'neg_mean_squared_log_error',
    'mean_squared_log_error': 'neg_mean_squared_log_error',
    'rmse': 'neg_root_mean_squared_error',
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'root_mean_squared_error': 'neg_root_mean_squared_error',
    'mean_absolute_percentage_error': 'mean_absolute_percentage_error',
    'r2': 'r2',
    'r2_score': 'r2',
    'logloss': 'neg_log_loss',
    'log_loss': 'neg_log_loss',
    'mape': mape,
    'smape': smape,
    'msle': msle,
    # ...
}

greater_is_better = {
    'mse': False,
    'mae': False,
    'rmse': False,
    'mape': False,
    'smape': False,
    'msle': False,
    'r2_score': True,
    'explained_variance_score': True,
    'max_error': False,
    'mean_absolute_error': False,
    'mean_squared_error': False,
    'mean_squared_log_error': False,
    'median_absolute_error': False,
    'mean_absolute_percentage_error': False,
    'mean_pinball_loss': False,
    'mean_tweedie_deviance': False,
    'mean_poisson_deviance': False,
    'mean_gamma_deviance': False,

    'accuracy_score': True,
    'balanced_accuracy_score': True,
    'top_k_accuracy': True,
    'roc_auc': True,
    # ...
}

def metric_to_scorer(metric, task, pos_label=None, **options):
    if pos_label is not None:
        options['pos_label'] = pos_label
    options['average'] = _task_to_average(task)

    if isinstance(metric, str) and isinstance(metric2scoring[metric], str):
        return get_scorer(metric2scoring[metric])
    elif isinstance(metric, str) and callable(metric2scoring[metric]):
        options.update({'greater_is_better': greater_is_better[metric]})
        return make_scorer(metric2scoring[metric], **options)
    elif callable(metric) and metric.__name__ in metric2scoring.keys():
        if isinstance(metric2scoring[metric.__name__], str):
            return get_scorer(metric2scoring[metric.__name__])
        else:
            options.update({'greater_is_better': greater_is_better[metric.__name__]})
            return make_scorer(metric2scoring[metric.__name__], **options)
    elif callable(metric) and metric.__name__ not in metric2scoring.keys():
        if options.get('optimize_direction') is not None:
            options.update({'greater_is_better': True
                    if options.get('optimize_direction') == 'max' else False})
            return make_scorer(metric, **options)
        else:
            raise ValueError('Note that custom reward_metric need to provide '
                             'optimize_direction.')
    else:
        raise ValueError('The reward_metric definition might be wrong.')

class Metrics(metrics.Metrics):
    calc_score = calc_score
    metric_to_scorer = metric_to_scorer