# -*- coding:utf-8 -*-
"""

"""
from hypernets.utils import logging
from hypernets.core.search_space import ModuleSpace

from hyperts.utils import consts
from hyperts.framework.wrappers.stats_wrappers import (ProphetWrapper,
                                                       VARWrapper,
                                                       ARIMAWrapper,
                                                       TSFWrapper)

logger = logging.get_logger(__name__)


class HyperEstimator(ModuleSpace):

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
        if yearly_seasonality is not None and changepoint_range != 'auto':
            kwargs['changepoint_range'] = changepoint_range
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
        if task == consts.Task_UNIVARIABLE_FORECAST:
            prophet = ProphetWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Prophet model supports only univariate forecast task.')
        return prophet


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
    Parameter Description Reference: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/arima/model.py

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
        if task == consts.Task_UNIVARIABLE_FORECAST:
            var = ARIMAWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('ARIMA model supports only univariate forecast task.')
        return var


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
    Parameter Description Reference: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/vector_ar/var_model.py
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
        if task == consts.Task_MULTIVARIABLE_FORECAST:
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
    Parameter Description Reference: https://github.com/alan-turing-institute/sktime/blob/main/sktime/classification/interval_based/_tsf.py
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
        if task in [consts.Task_UNIVARIABLE_BINARYCLASS, consts.Task_UNIVARIABLE_MULTICALSS]:
            tsf = TSFWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('TSF model supports only univariable classification task.')
        return tsf