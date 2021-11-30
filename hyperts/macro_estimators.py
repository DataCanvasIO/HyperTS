# -*- coding:utf-8 -*-
"""

"""
from hypernets.utils import logging
from hypernets.core.search_space import ModuleSpace

from hyperts.utils import consts
from hyperts.wrappers.stats_wrappers import ProphetWrapper, VARWrapper, TSFClassifierWrapper

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
            kwargs['additive'] = 'additive'
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
        if task == consts.TASK_UNIVARIABLE_FORECAST:
            prophet = ProphetWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('Prophet model supports only univariate forecast task.')
        return prophet


class VARForecastEstimator(HyperEstimator):
    """Time Series Forecast Estimator based on Hypernets.
    Estimator: Vector Autoregression (VAR).
    Suitable for: Multivariate Forecast Task.
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
        if task == consts.TASK_MULTIVARIABLE_FORECAST:
            var = VARWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('VAR model supports only multivariate forecast task.')
        return var


class TSFClassificationEstimator(HyperEstimator):
    """Time Series Classfication Estimator based on Hypernets.
    Estimator: Vector Autoregression (VAR).
    Suitable for: Classfication Task.
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
        if task in [consts.TASK_BINARY_CLASSIFICATION, consts.TASK_MULTICLASS_CLASSIFICATION]:
            tsf = TSFClassifierWrapper(fit_kwargs, **kwargs)
        else:
            raise ValueError('TSF model supports only classification task.')
        return tsf









