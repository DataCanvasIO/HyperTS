# -*- coding:utf-8 -*-
"""

"""
from hypernets.utils import logging
from hypernets.core.search_space import ModuleSpace

from hyperts.utils import consts

from hyperts.framework.wrappers.stats_wrappers import ProphetWrapper
from hyperts.framework.wrappers.stats_wrappers import is_prophet_installed
from hyperts.framework.wrappers.stats_wrappers import VARWrapper
from hyperts.framework.wrappers.stats_wrappers import ARIMAWrapper
from hyperts.framework.wrappers.stats_wrappers import TSForestWrapper
from hyperts.framework.wrappers.stats_wrappers import KNeighborsWrapper
from hyperts.framework.wrappers.stats_wrappers import IForestWrapper
from hyperts.framework.wrappers.stats_wrappers import OneClassSVMWrapper

try:
    import tensorflow
except:
    is_tensorflow_installed = False
else:
    is_tensorflow_installed = True

if is_tensorflow_installed:
    from hyperts.framework.wrappers.dl_wrappers import DeepARWrapper
    from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper
    from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper
    from hyperts.framework.wrappers.dl_wrappers import NBeatsWrapper
    from hyperts.framework.wrappers.dl_wrappers import InceptionTimeWrapper
    from hyperts.framework.wrappers.dl_wrappers import ConvVAEWrapper


logger = logging.get_logger(__name__)


##################################### Define Statistic Model HyperEstimator #####################################
class HyperEstimator(ModuleSpace):
    """An interface class representing a hyperts estimator.

    """
    def __init__(self, fit_kwargs=None, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.estimator = None

    def _build_estimator(self, task, fit_kwargs, kwargs):
        raise NotImplementedError

    def build_estimator(self, task=None):
        pv = self.param_values
        self.estimator = self._build_estimator(task, self.fit_kwargs, pv)

    def _compile(self):
        pass

    def _forward(self, inputs):
        return self.estimator


class ProphetForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Prophet.
    Suitable for: Unvariate Forecast Task.

    Parameters
    ----------
    growth: String 'linear' or 'logistic' to specify a linear or logistic
        trend.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: Number of simulated draws used to estimate
        uncertainty intervals. Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.

    Notes
    ----------
    Parameter Description Reference: https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py
    """

    def __init__(self, fit_kwargs=None, growth='linear', changepoints=None,
                 n_changepoints=25, changepoint_range=0.8, yearly_seasonality='auto',
                 weekly_seasonality='auto', daily_seasonality='auto', holidays=None,
                 seasonality_mode='additive', seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0, changepoint_prior_scale=0.05,
                 mcmc_samples=0, interval_width=0.80, uncertainty_samples=1000,
                 space=None, name=None, **kwargs):

        if growth is not None and growth != 'linear':
            kwargs['growth'] = growth
        if changepoints is not None:
            kwargs['changepoints'] = changepoints
        if n_changepoints is not None and n_changepoints != 25:
            kwargs['n_changepoints'] = n_changepoints
        if changepoint_range is not None and changepoint_range != 0.8:
            kwargs['changepoint_range'] = changepoint_range
        if yearly_seasonality is not None and yearly_seasonality != 'auto':
            kwargs['yearly_seasonality'] = yearly_seasonality
        if weekly_seasonality is not None and weekly_seasonality != 'auto':
            kwargs['weekly_seasonality'] = weekly_seasonality
        if daily_seasonality is not None and daily_seasonality != 'auto':
            kwargs['daily_seasonality'] = daily_seasonality
        if holidays is not None:
            kwargs['holidays'] = holidays
        if seasonality_mode is not None and seasonality_mode != 'additive':
            kwargs['seasonality_mode'] = seasonality_mode
        if seasonality_prior_scale is not None and seasonality_prior_scale != 10.0:
            kwargs['seasonality_prior_scale'] = seasonality_prior_scale
        if holidays_prior_scale is not None and holidays_prior_scale != 10.0:
            kwargs['holidays_prior_scale'] = holidays_prior_scale
        if changepoint_prior_scale is not None and changepoint_prior_scale != 0.05:
            kwargs['changepoint_prior_scale'] = changepoint_prior_scale
        if mcmc_samples is not None and mcmc_samples != 0:
            kwargs['mcmc_samples'] = mcmc_samples
        if interval_width is not None and interval_width != 0.80:
            kwargs['interval_width'] = interval_width
        if uncertainty_samples is not None and uncertainty_samples != 1000:
            kwargs['uncertainty_samples'] = uncertainty_samples

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task == consts.Task_UNIVARIATE_FORECAST:
            prophet = ProphetWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Prophet model supports only univariate forecast task.')
        return prophet

    @property
    def is_prophet_installed(self):
        if is_prophet_installed:
            return True
        else:
            return False


class ARIMAForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Autoregressive Integrated Moving Average (ARIMA).
    Suitable for: Univariate Forecast Task.

    Parameters
    ----------
    p: autoregressive order.
    q: moving average order.
    d: differences order.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend. Can be specified as a
        string where 'c' indicates a constant term, 't' indicates a
        linear trend in time, and 'ct' includes both. Can also be specified as
        an iterable defining a polynomial, as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is 'c' for
        models without integration, and no trend for models with integration.

    Notes
    ----------
    Parameter Description Reference: https://github.com/statsmodels/statsmodels/blob/main/
        statsmodels/tsa/arima/model.py

    The (p,d,q) order of the model for the autoregressive, differences, and
    moving average components. d is always an integer, while p and q may
    either be integers or lists of integers.

    - autoregressive models: AR(p)
    - moving average models: MA(q)
    - mixed autoregressive moving average models: ARMA(p, q)
    - integration models: ARIMA(p, d, q)
    """

    def __init__(self, fit_kwargs=None,
                 p=1, d=0, q=0, seasonal_order=(0, 0, 0, 0), trend='c',
                 space=None, name=None, **kwargs):

        if p is not None and p != 1:
            kwargs['p'] = p
        if d is not None and d != 0:
            kwargs['d'] = d
        if q is not None and q != 0:
            kwargs['q'] = q
        if seasonal_order is not None and seasonal_order != (0, 0, 0, 0):
            kwargs['seasonal_order'] = seasonal_order
        if trend is not None and trend != 'c':
            kwargs['trend'] = trend

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task == consts.Task_UNIVARIATE_FORECAST:
            arima = ARIMAWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('ARIMA model supports only univariate forecast task.')
        return arima


class VARForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Vector Autoregression (VAR).
    Suitable for: Multivariate Forecast Task.

    Parameters
    ----------
    maxlags : {int, None}, default None
        Maximum number of lags to check for order selection, defaults to
        12 * (nobs/100.)**(1./4), see select_order function
    method : {'ols'}
        Estimation method to use
    ic : {'aic', 'fpe', 'hqic', 'bic', None}
        Information criterion to use for VAR order selection.
        aic : Akaike
        fpe : Final prediction error
        hqic : Hannan-Quinn
        bic : Bayesian a.k.a. Schwarz
    trend : str {"c", "ct", "ctt", "nc", "n"}
        "c" - add constant
        "ct" - constant and trend
        "ctt" - constant, linear and quadratic trend
        "n", "nc" - co constant, no trend
        Note that these are prepended to the columns of the dataset.

    Notes
    ----------
    Parameter Description Reference: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/
        tsa/vector_ar/var_model.py
    """

    def __init__(self, fit_kwargs=None, maxlags=None,
                 method='ols', ic=None, trend='c',
                 space=None, name=None, **kwargs):

        if maxlags is not None:
            kwargs['maxlags'] = maxlags
        if method is not None and method != 'ols':
            kwargs['method'] = method
        if ic is not None:
            kwargs['ic'] = ic
        if trend is not None and trend != 'c':
            kwargs['trend'] = trend

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task == consts.Task_MULTIVARIATE_FORECAST:
            var = VARWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('VAR model supports only multivariate forecast task.')
        return var


class TSFClassificationEstimator(HyperEstimator):
    """Time Series Classfication Estimator based on Hypernets.
    Estimator: Time Series Forest (TSF).
    Suitable for: Classfication Task.

    Parameters
    ----------
    n_estimators : int, ensemble size, optional (default = 200)
    min_interval : int, minimum width of an interval, optional (default to 3)
    n_jobs : int, optional (default=1) The number of jobs to run in parallel for
        both `fit` and `predict`.  ``-1`` means using all processors.
    random_state : int, seed for random, optional (default = none)

    Notes
    ----------
    Parameter Description Reference: https://github.com/alan-turing-institute/sktime/blob/main/sktime/
        classification/interval_based/_tsf.py
    """

    def __init__(self, fit_kwargs=None, min_interval=3,
                 n_estimators=200, n_jobs=1, random_state=None,
                 space=None, name=None, **kwargs):

        if min_interval is not None and min_interval != 3:
            kwargs['min_interval'] = min_interval
        if n_estimators is not None and n_estimators != 200:
            kwargs['n_estimators'] = n_estimators
        if n_jobs is not None and n_jobs != 1:
            kwargs['n_jobs'] = n_jobs
        if random_state is not None:
            kwargs['random_state'] = random_state

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in [consts.Task_UNIVARIATE_BINARYCLASS, consts.Task_UNIVARIATE_MULTICALSS]:
            tsf = TSForestWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('TSF model supports only univariate classification task.')
        return tsf


class KNNClassificationEstimator(HyperEstimator):
    """Time Series Classfication Estimator based on Hypernets.
    Estimator: K Nearest Neighbors (KNN).
    Suitable for: Classfication Task.

    Parameters
    ----------
    n_neighbors     : int, set k for knn (default =1)
    weights         : string or callable function, optional, default =='uniform'
                      mechanism for weighting a vote, one of: 'uniform', 'distance'
                      or a callable function
    algorithm       : search method for neighbours {'auto', 'ball_tree',
                      'kd_tree', 'brute'}: default = 'brute'
    distance        : distance measure for time series: {'dtw','ddtw',
                      'wdtw','lcss','erp','msm','twe'}: default ='dtw'

    Notes
    ----------
    Parameter Description Reference: https://github.com/alan-turing-institute/sktime/blob/main/sktime/
        classification/distance_based/_time_series_neighbors.py
    """

    def __init__(self, fit_kwargs=None, n_neighbors=1,
                 weights='uniform', algorithm='brute', distance='dtw',
                 space=None, name=None, **kwargs):

        if n_neighbors is not None and n_neighbors != 1:
            kwargs['n_neighbors'] = n_neighbors
        if weights is not None and weights != 'uniform':
            kwargs['weights'] = weights
        if algorithm is not None and algorithm != 'brute':
            kwargs['algorithm'] = algorithm
        if distance is not None and distance != 'dtw':
            kwargs['distance'] = distance

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_CLASSIFICATION:
            knn = KNeighborsWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('KNN model supports only classification task.')
        return knn


class IForestDetectionEstimator(HyperEstimator):
    """Time Series Anomaly Detection Estimator based on Hypernets.
    Estimator:  Isolation Forest (IForest).
    Suitable for: Univariate/Multivariate Anomaly Detection Task.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).
    contamination : 'auto' or float, default=0.05
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    verbose : int, default=0
        Controls the verbosity of the tree building process.
    """
    def __init__(self, fit_kwargs=None, n_estimators=100,
                 max_samples="auto", contamination=0.05, max_features=1.0,
                 bootstrap=False, n_jobs=None, random_state=None,
                 verbose=0, space=None, name=None, **kwargs):

        if n_estimators is not None and n_estimators != 100:
            kwargs['n_estimators'] = n_estimators
        if max_samples is not None and max_samples != 'auto':
            kwargs['max_samples'] = max_samples
        if contamination is not None and contamination != 0.05:
            kwargs['contamination'] = contamination
        if max_features is not None and max_features != 1.0:
            kwargs['max_features'] = max_features
        if bootstrap is not None and bootstrap != False:
            kwargs['bootstrap'] = bootstrap
        if n_jobs is not None:
            kwargs['n_jobs'] = n_jobs
        if random_state is not None:
            kwargs['random_state'] = random_state
        if verbose is not None and verbose != 0:
            kwargs['verbose'] = verbose

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_DETECTION:
            iforest = IForestWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Isolation Forest model supports only anomaly detection task.')
        return iforest


class OCSVMDetectionEstimator(HyperEstimator):
    """Time Series Anomaly Detection Estimator based on Hypernets.
    Estimator:  One Class SVM (OCSVM).
    Suitable for: Univariate/Multivariate Anomaly Detection Task.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.
    degree : int, default=2
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.
        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    contamination : 'auto' or float, default=0.05
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    """
    def __init__(self, fit_kwargs=None, kernel="rbf", degree=2,
                 gamma="auto", coef0=0.0, tol=1e-3, nu=0.5, shrinking=True,
                 cache_size=200, max_iter=-1, contamination=0.05, verbose=False,
                 space=None, name=None, **kwargs):

        if kernel is not None and kernel != 'rbf':
            kwargs['kernel'] = kernel
        if degree is not None and degree != 2:
            kwargs['degree'] = degree
        if gamma is not None and gamma != 'auto':
            kwargs['gamma'] = gamma
        if coef0 is not None and coef0 != 0.0:
            kwargs['coef0'] = coef0
        if tol is not None and tol != 1e-3:
            kwargs['tol'] = tol
        if nu is not None and nu != 0.5:
            kwargs['nu'] = nu
        if shrinking is not None and shrinking != True:
            kwargs['shrinking'] = shrinking
        if cache_size is not None and cache_size !=200:
            kwargs['cache_size'] = cache_size
        if max_iter is not None and max_iter != -1:
            kwargs['max_iter'] = max_iter
        if contamination is not None and contamination != 0.05:
            kwargs['contamination'] = contamination
        if verbose is not None and verbose != False:
            kwargs['verbose'] = verbose

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_DETECTION:
            ocsvm = OneClassSVMWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('OneClassSVM model supports only anomaly detection task.')
        return ocsvm


##################################### Define Deep Learning Model HyperEstimator #####################################
class DeepARForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Deep AutoRegressive (DeepAR).
    Suitable for: Univariate Forecast Task.

    Parameters
    ----------
    timestamp  : Str - Timestamp name, not optional.
    task       : Str - Only 'univariate-forecast' is supported,
                 default = 'univariate-forecast'.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'basic', 'gru', 'lstm}, default = 'gru'.
    rnn_units  : Positive Int - The dimensionality of the output space for recurrent neural network,
                 default = 16.
    rnn_layers : Positive Int - The number of the layers for recurrent neural network,
                 default = 1.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid', 'tanh'},
                 default = 'linear'.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets,
                 default = 0.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 3.
    horizon    : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor    : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Only 'log_gaussian_loss' is supported for DeepAR, which has been defined.
                 default = 'log_gaussian_loss'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update.
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.
    """

    def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-forecast',
                 rnn_type='gru', rnn_units=16, rnn_layers=1, out_activation='linear',
                 drop_rate=0., window=3, horizon=1, forecast_length=1, metrics='auto',
                 monitor='val_loss', optimizer='auto', learning_rate=0.001, loss='log_gaussian_loss',
                 reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):

        if timestamp is not None:
            kwargs['timestamp'] = timestamp
        else:
            raise ValueError('timestamp can not be None.')
        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if rnn_type is not None and rnn_type != 'gru':
            kwargs['rnn_type'] = rnn_type
        if rnn_units is not None and rnn_units != 16:
            kwargs['rnn_units'] = rnn_units
        if rnn_layers is not None and rnn_layers != 1:
            kwargs['rnn_layers'] = rnn_layers
        if drop_rate is not None and drop_rate != 0.:
            kwargs['drop_rate'] = drop_rate
        if out_activation is not None and out_activation != 'linear':
            kwargs['out_activation'] = out_activation
        if window is not None and window != 3:
            kwargs['window'] = window
        if horizon is not None and horizon != 1:
            kwargs['horizon'] = horizon
        if forecast_length is not None and forecast_length != 1:
            kwargs['forecast_length'] = forecast_length
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor is not None and monitor != 'val_loss':
            kwargs['monitor'] = monitor
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if loss is not None and loss != 'log_gaussian_loss':
            kwargs['loss'] = loss
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task == consts.Task_UNIVARIATE_FORECAST:
            deepar = DeepARWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('DeepAR model supports only univariate forecast task.')
        return deepar


class HybirdRNNGeneralEstimator(HyperEstimator):
    """Time Series Forecast|Classification|Regression Estimator based on Hypernets.
    Estimator: SimpleRNN|GRU|LSTM (HybirdRNN).
    Suitable for: The General Time Series Tasks.

    Parameters
    ----------
    timestamp  : Str or None - Timestamp name, the forecast task must be given.
    task       : Str - Support forecast, classification, and regression.
                 default = 'univariate-forecast'.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'basic', 'gru', 'lstm}, default = 'gru'.
    rnn_units  : Positive Int - The dimensionality of the output space for recurrent neural network,
                 default = 16.
    rnn_layers : Positive Int - The number of the layers for recurrent neural network,
                 default = 1.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid', 'tanh'},
                 default = 'linear'.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets,
                 default = 0.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 3.
    horizon    : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor    : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Loss function, optional {'auto', 'adam', 'sgd'},
                 default = 'auto'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update.
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.
    """

    def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-forecast',
                 rnn_type='gru', rnn_units=16, rnn_layers=1, out_activation='linear',
                 drop_rate=0., window=3, horizon=1, forecast_length=1, metrics='auto',
                 monitor='val_loss', optimizer='auto', learning_rate=0.001, loss='auto',
                 reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):

        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if task in consts.TASK_LIST_FORECAST and timestamp is None:
            raise ValueError('Timestamp need to be given for forecast task.')
        else:
            kwargs['timestamp'] = timestamp
        if rnn_type is not None and rnn_type != 'gru':
            kwargs['rnn_type'] = rnn_type
        if rnn_units is not None and rnn_units != 16:
            kwargs['rnn_units'] = rnn_units
        if rnn_layers is not None and rnn_layers != 1:
            kwargs['rnn_layers'] = rnn_layers
        if drop_rate is not None and drop_rate != 0.:
            kwargs['drop_rate'] = drop_rate
        if out_activation is not None and out_activation != 'linear':
            kwargs['out_activation'] = out_activation
        if window is not None and window != 7:
            kwargs['window'] = window
        if horizon is not None and horizon != 1:
            kwargs['horizon'] = horizon
        if forecast_length is not None and forecast_length != 1:
            kwargs['forecast_length'] = forecast_length
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor is not None and monitor != 'val_loss':
            kwargs['monitor'] = monitor
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if loss is not None and loss != 'auto':
            kwargs['loss'] = loss
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
            rnn = HybirdRNNWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Check whether the task type meets specifications.')
        return rnn


class LSTNetGeneralEstimator(HyperEstimator):
    """Time Series Forecast|Classification|Regression Estimator based on Hypernets.
    Estimator: Long-and Short-term Time-series network (LSTNet).
    Suitable for: The General Time Series Tasks.

    Parameters
    ----------
    timestamp  : Str or None - Timestamp name, the forecast task must be given.
    task       : Str - Support forecast, classification, and regression.
                 default = 'univariate-forecast'.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'basic', 'gru', 'lstm}, default = 'gru'.
    skip_rnn_type : Str - Type of skip recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}, default = 'gru'.
    cnn_filters: Positive Int - The dimensionality of the output space (i.e. the number of filters
                 in the convolution), default = 16.
    kernel_size: Positive Int - A single integer specifying the spatial dimensions of the filters,
                 default = 1.
    rnn_units  : Positive Int - The dimensionality of the output space for recurrent neural network,
                 default = 16.
    rnn_layers : Positive Int - The number of the layers for recurrent neural network,
                 default = 1.
    skip_rnn_units : Positive Int - The dimensionality of the output space for skip recurrent neural network,
                 default = 16.
    skip_rnn_layers : Positive Int - The number of the layers for skip recurrent neural network,
                 default = 1.
    skip_period: Positive Int or None - The length of skip for recurrent neural network,
                 default = None.
    ar_order   : Positive Int or None - The window size of the autoregressive component,
                 default = None.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets,
                 default = 0.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid', 'tanh'},
                 default = 'linear'.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 7.
    horizon    : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor    : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Loss function, optional {'auto', 'adam', 'sgd'},
                 default = 'auto'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update,
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.
    """

    def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-forecast',
                 rnn_type='gru', skip_rnn_type='gru',
                 cnn_filters=16, kernel_size=1, rnn_units=16, rnn_layers=1,
                 skip_rnn_units=16, skip_rnn_layers=1, skip_period=0, ar_order=0,
                 drop_rate=0., out_activation='linear', window=7, horizon=1, forecast_length=1,
                 metrics='auto', monitor='val_loss', optimizer='auto', learning_rate=0.001,
                 loss='auto', reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):

        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if task in consts.TASK_LIST_FORECAST and timestamp is None:
            raise ValueError('Timestamp need to be given for forecast task.')
        else:
            kwargs['timestamp'] = timestamp
        if rnn_type is not None and rnn_type != 'gru':
            kwargs['rnn_type'] = rnn_type
        if skip_rnn_type is not None and skip_rnn_type != 'gru':
            kwargs['skip_rnn_type'] = skip_rnn_type
        if cnn_filters is not None and cnn_filters != 16:
            kwargs['cnn_filters'] = cnn_filters
        if kernel_size is not None and kernel_size != 1:
            kwargs['kernel_size'] = kernel_size
        if rnn_units is not None and rnn_units != 16:
            kwargs['rnn_units'] = rnn_units
        if rnn_layers is not None and rnn_layers != 1:
            kwargs['rnn_layers'] = rnn_layers
        if skip_rnn_units is not None and skip_rnn_units != 16:
            kwargs['skip_rnn_units'] = skip_rnn_units
        if skip_rnn_layers is not None and skip_rnn_layers != 1:
            kwargs['skip_rnn_layers'] = skip_rnn_layers
        if skip_period is not None and skip_period != 0:
            kwargs['skip_period'] = skip_period
        if ar_order is not None and ar_order != 0:
            kwargs['ar_order'] = ar_order
        if drop_rate is not None and drop_rate != 0.:
            kwargs['drop_rate'] = drop_rate
        if out_activation is not None and out_activation != 'linear':
            kwargs['out_activation'] = out_activation
        if window is not None and window != 7:
            kwargs['window'] = window
        if horizon is not None and horizon != 1:
            kwargs['horizon'] = horizon
        if forecast_length is not None and forecast_length != 1:
            kwargs['forecast_length'] = forecast_length
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor is not None and monitor != 'val_loss':
            kwargs['monitor'] = monitor
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if loss is not None and loss != 'auto':
            kwargs['loss'] = loss
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
            lstnet = LSTNetWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Check whether the task type meets specifications.')
        return lstnet


class NBeatsForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting (NBeats).
    Suitable for: Univariate/Multivariate Forecast Task.

    Parameters
    ----------
    timestamp  : Str - Timestamp name, not optional.
    task       : Str - Only 'forecast' is supported,
                 default = 'univariate-forecast'.
    stack_types : Tuple(Str) - Stack types, optional {'trend', 'seasonality', generic}.
                  default = ('trend', 'seasonality').
    thetas_dim  : Tuple(Int) - The number of units that make up each dense layer in each block of every stack.
                  default = (4, 8).
    nb_blocks_per_stack : Int - The number of block per stack.
                  default = 3.
    share_weights_in_stack : Bool - Whether to share weights in stack.
                  default = False.
    hidden_layer_units : Int - The units of hidden layer.
                  default = 256.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid', 'tanh'},
                 default = 'linear'.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 3.
    horizon    : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor    : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Only 'log_gaussian_loss' is supported for DeepAR, which has been defined.
                 default = 'log_gaussian_loss'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update.
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.
    """

    def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-forecast',
                 stack_types=('trend', 'seasonality'), thetas_dim=(4, 8), nb_blocks_per_stack=3,
                 share_weights_in_stack=False, hidden_layer_units=256, out_activation='linear',
                 window=3, horizon=1, forecast_length=1, metrics='auto',
                 monitor='val_loss', optimizer='auto', learning_rate=0.001, loss='auto',
                 reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):

        if timestamp is not None:
            kwargs['timestamp'] = timestamp
        else:
            raise ValueError('timestamp can not be None.')
        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if stack_types is not None and stack_types != ('trend', 'seasonality'):
            kwargs['stack_types'] = stack_types
        if thetas_dim is not None and thetas_dim != (4, 8):
            kwargs['thetas_dim'] = thetas_dim
        if nb_blocks_per_stack is not None and nb_blocks_per_stack != 3:
            kwargs['nb_blocks_per_stack'] = nb_blocks_per_stack
        if share_weights_in_stack is not None and share_weights_in_stack != False:
            kwargs['share_weights_in_stack'] = share_weights_in_stack
        if hidden_layer_units is not None and hidden_layer_units != 256:
            kwargs['hidden_layer_units'] = hidden_layer_units
        if out_activation is not None and out_activation != 'linear':
            kwargs['out_activation'] = out_activation
        if window is not None and window != 3:
            kwargs['window'] = window
        if horizon is not None and horizon != 1:
            kwargs['horizon'] = horizon
        if forecast_length is not None and forecast_length != 1:
            kwargs['forecast_length'] = forecast_length
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor is not None and monitor != 'val_loss':
            kwargs['monitor'] = monitor
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if loss is not None and loss != 'auto':
            kwargs['loss'] = loss
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_FORECAST:
            nbeats = NBeatsWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('NBeats model supports only forecast task.')
        return nbeats


class InceptionTimeGeneralEstimator(HyperEstimator):
    """Time Series Classification or Regression Estimator based on Hypernets.
    Estimator:  Inception Time (InceptionTime).
    Suitable for: Univariate/Multivariate Classification or Regression Task.

    Parameters
    ----------
    timestamp  : Str - Timestamp name, not optional.
    task       : Str - Only 'classification' is supported,
                 default = 'univariate-binaryclass'.
    blocks     : Int - The depth of the net architecture.
                 default = 3.
    cnn_filters: Int - The number of cnn filters.
                 default = 32.
    bottleneck_size: Int - The number of bottleneck (a cnn layer).
                 default = 32.
    kernel_size_list: Tuple - The kernel size of cnn for a inceptionblock.
                 default = (1, 3, 5, 8, 12).
    shortcut   : Bool - Whether to use shortcut opration.
                 default = True.
    short_filters: Int - The number of filters of shortcut conv1d layer.
                 default = 64.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor    : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Only 'log_gaussian_loss' is supported for DeepAR, which has been defined.
                 default = 'log_gaussian_loss'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update.
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.
    """

    def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-binaryclass',
                 blocks=3, cnn_filters=32, kernel_size_list=(1, 3, 5, 8, 12),
                 bottleneck_size=32, shortcut=True, short_filters=64, metrics='auto',
                 monitor='val_loss', optimizer='auto', learning_rate=0.001, loss='auto',
                 reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):
        kwargs['timestamp'] = timestamp
        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if blocks is not None and blocks != 3:
            kwargs['blocks'] = blocks
        if cnn_filters is not None and cnn_filters != 32:
            kwargs['cnn_filters'] = cnn_filters
        if kernel_size_list is not None and kernel_size_list != (1, 3, 5, 8, 12):
            kwargs['kernel_size_list'] = kernel_size_list
        if bottleneck_size is not None and bottleneck_size != 32:
            kwargs['bottleneck_size'] = bottleneck_size
        if shortcut is not None and shortcut != True:
            kwargs['shortcut'] = shortcut
        if short_filters is not None and short_filters != 64:
            kwargs['short_filters'] = short_filters
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor is not None and monitor != 'val_loss':
            kwargs['monitor'] = monitor
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if loss is not None and loss != 'auto':
            kwargs['loss'] = loss
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
            inceptiontime = InceptionTimeWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('InceptionTime model supports only classification or regression task.')
        return inceptiontime


class ConvVAEDetectionEstimator(HyperEstimator):
    """Time Series Anomaly Detection Estimator based on Hypernets.
    Estimator:  Convolution Variational AutoEncoder (ConvVAE).
    Suitable for: Univariate/Multivariate Anomaly Detection Task.

    Parameters
    ----------
    task          : Str - Only support anomaly detection.
                See hyperts.utils.consts for details.
    timestamp     : Str or None - Timestamp name, the forecast task must be given,
                default None.
    window        : Positive Int - Length of the time series sequences for a sample.
    horizon       : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    latent_dim    : Int - Latent representation of encoder, default 2.
    conv_type     : Str - Type of 1D convolution, optional {'general', 'separable'},
                default 'general'.
    cnn_filters   : Positive Int - The dimensionality of the output space (i.e. the number
        of filters in the convolution).
    kernel_size   : Positive Int - A single integer specifying the spatial dimensions
        of the filters,
    strides       : Int or tuple/list of a single integer - Specifying the stride length
        of the convolution.
    nb_layers     : Int - The layers of encoder and decoder, default 2.
    activation    : Str - The activation of hidden layers, default 'relu'.
    drop_rate     : Float between 0 and 1 - The rate of Dropout for neural nets.
    out_activation : Str - Forecast the task output activation function,
                 optional {'linear', 'sigmoid', 'tanh'}, default 'linear'.
    metrics       : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor_metric : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer     : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary       : Bool - Whether to output network structure information,
                 default = True.
    batch_size : Int or None - Number of samples per gradient update.
                 default = 32.
    epochs     : Int - Number of epochs to train the model,
                 default = 1.
    verbose    : 0, 1, or 2. Verbosity mode.
                 0 = silent, 1 = progress bar, 2 = one line per epoch.
                 Note that the progress bar is not particularly useful when logged to a file, so verbose=2
                 is recommended when not running interactively (eg, in a production environment).
                 default = 1.
    callbacks  : List of `keras.callbacks.Callback` instances.
                 List of callbacks to apply during training.
                 See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
                 and `tf.keras.callbacks.History` callbacks are created automatically
                 and need not be passed into `model.fit`.
                 `tf.keras.callbacks.ProgbarLogger` is created or not based on
                 `verbose` argument to `model.fit`.
                 default = None.
    validation_split : Float between 0 and 1.
                 Fraction of the training data to be used as validation data.
                 The model will set apart this fraction of the training data, will not train on it, and will
                 evaluate the loss and any model metrics on this data at the end of each epoch,
                 default = 0.
    shuffle    : Boolean (whether to shuffle the training data
                 before each epoch) or str (for 'batch').
    max_queue_size : Int - Used for generator or `keras.utils.Sequence`
                 input only. Maximum size for the generator queue,
                 default = 10.
    workers    : Int - Used for generator or `keras.utils.Sequence` input
                 only. Maximum number of processes to spin up when using process-based
                 threading. If 0, will execute the generator on the main thread,
                 default = 1.
    use_multiprocessing : Bool. Used for generator or
                 `keras.utils.Sequence` input only. If `True`, use process-based
                 threading. Note that because this implementation relies on
                 multiprocessing, you should not pass non-picklable arguments to
                 the generator as they can't be passed easily to children processes.
                 default = False.

    """
    def __init__(self, fit_kwargs=None, timestamp=None, task='detection',
                 contamination=0.05, window=3, horizon=1, forecast_length=1,
                 latent_dim=2, conv_type='general', cnn_filters=16, kernel_size=1,
                 strides=1, nb_layers=2, activation='relu', drop_rate=0.2,
                 out_activation='linear', reconstract_dim=None, metrics='auto',
                 monitor_metric='val_loss', optimizer='auto', learning_rate=0.001,
                 reducelr_patience=5, earlystop_patience=10, summary=True,
                 batch_size=None, epochs=1, verbose=1, callbacks=None,
                 validation_split=0., shuffle=True, max_queue_size=10,
                 workers=1, use_multiprocessing=False,
                 space=None, name=None, **kwargs):
        if timestamp is not None:
            kwargs['timestamp'] = timestamp
        else:
            raise ValueError('timestamp can not be None.')
        if task is not None:
            kwargs['task'] = task
        else:
            raise ValueError('task can not be None.')
        if contamination is not None and contamination != 0.05:
            kwargs['contamination'] = contamination
        if window is not None and window != 3:
            kwargs['window'] = window
        if horizon is not None and horizon != 1:
            kwargs['horizon'] = horizon
        if forecast_length is not None and forecast_length != 1:
            kwargs['forecast_length'] = forecast_length
        if latent_dim is not None and latent_dim != 2:
            kwargs['latent_dim'] = latent_dim
        if conv_type is not None and conv_type != 'general':
            kwargs['conv_type'] = conv_type
        if cnn_filters is not None and cnn_filters != 16:
            kwargs['cnn_filters'] = cnn_filters
        if kernel_size is not None and kernel_size != 1:
            kwargs['kernel_size'] = kernel_size
        if strides is not None and strides != 1:
            kwargs['strides'] = strides
        if nb_layers is not None and nb_layers != 2:
            kwargs['nb_layers'] = nb_layers
        if activation is not None and activation != 'relu':
            kwargs['activation'] = activation
        if drop_rate is not None and drop_rate != 0.2:
            kwargs['drop_rate'] = drop_rate
        if out_activation is not None and out_activation != 'linear':
            kwargs['out_activation'] = out_activation
        if reconstract_dim is not None:
            kwargs['reconstract_dim'] = reconstract_dim
        if metrics is not None and metrics != 'auto':
            kwargs['metrics'] = metrics
        if monitor_metric is not None and monitor_metric != 'val_loss':
            kwargs['monitor_metric'] = monitor_metric
        if optimizer is not None and optimizer != 'auto':
            kwargs['optimizer'] = optimizer
        if learning_rate is not None and learning_rate != 0.001:
            kwargs['learning_rate'] = learning_rate
        if reducelr_patience is not None and reducelr_patience != 5:
            kwargs['reducelr_patience'] = reducelr_patience
        if earlystop_patience is not None and earlystop_patience != 10:
            kwargs['earlystop_patience'] = earlystop_patience
        if summary is not None and summary != True:
            kwargs['summary'] = summary

        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        if epochs is not None and epochs != 1:
            kwargs['epochs'] = epochs
        if verbose is not None and verbose != 1:
            kwargs['verbose'] = verbose
        if callbacks is not None:
            kwargs['callbacks'] = callbacks
        if validation_split is not None and validation_split != 0.:
            kwargs['validation_split'] = validation_split
        if shuffle is not None and shuffle != True:
            kwargs['shuffle'] = shuffle
        if max_queue_size is not None and max_queue_size != 10:
            kwargs['max_queue_size'] = max_queue_size
        if workers is not None and workers != 1:
            kwargs['workers'] = workers
        if use_multiprocessing is not None and use_multiprocessing != False:
            kwargs['use_multiprocessing'] = use_multiprocessing

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, fit_kwargs, kwargs):
        if task in consts.TASK_LIST_DETECTION:
            vae = ConvVAEWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('ConvVAE model supports only anomaly detection task.')
        return vae