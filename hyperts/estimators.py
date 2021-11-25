import numpy as np
try:
    from prophet import Prophet
except:
    from fbprophet import Prophet
from statsmodels.tsa.vector_ar.var_model import VAR
from sktime.classification.interval_based import TimeSeriesForestClassifier


from hypernets.core.search_space import ModuleSpace
from hypernets.utils import logging

from hyperts.utils import consts

logger = logging.get_logger(__name__)


class EstimatorWrapper:

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, periods):
        raise NotImplementedError


class ProphetWrapper(EstimatorWrapper):

    def __init__(self, fit_kwargs, **kwargs):
        self.timestamp = fit_kwargs.get('timestamp', consts.TIMESTAMP)
        self.model = Prophet(**kwargs)

    def fit(self, X, y):
        # adapt for prophet
        df_train = X[[self.timestamp]]
        if self.timestamp != 'ds':
            df_train.rename(columns={self.timestamp: 'ds'}, inplace=True)
        df_train['y'] = y
        self.model.fit(df_train)

    def predict(self, X):
        df_predict = self.model.predict(X)
        return df_predict['yhat'].values


class VARWrapper(EstimatorWrapper):

    def __init__(self, fit_kwargs, **kwargs):
        self.timestamp = fit_kwargs.get('timestamp', consts.TIMESTAMP)
        self.init_kwargs = kwargs if kwargs is not None else {}
        self.model = None

        # fitted
        self._end_date = None
        self._freq = None

    def fit(self, X, y):
        # adapt for prophet
        date_series_top2 = X[self.timestamp][:2].tolist()
        self._freq = (date_series_top2[1] - date_series_top2[0]).total_seconds()
        self._end_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()

        model = VAR(endog=y, dates=X[self.timestamp])
        self.model = model.fit(**self.init_kwargs)

    def predict(self, X):
        last_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()
        steps = int((last_date - self._end_date).total_seconds()/self._freq)
        predict_result = self.model.forecast(self.model.y, steps=steps)

        def calc_index(date):
            r_i = int((date - self._end_date).total_seconds()/self._freq)-1
            return predict_result[r_i].tolist()

        return np.array(X[self.timestamp].map(calc_index).to_list())


class SKTimeWrapper(EstimatorWrapper):

    def __init__(self, fit_kwargs, **kwargs):
        self.model = TimeSeriesForestClassifier(**kwargs)

    def fit(self, X, y):
        # adapt for prophet
        self.model.fit(X, y)

    def predict(self, X):
        predict_result = self.model.predict(X)
        return predict_result


class TSEstimatorMS(ModuleSpace):
    def __init__(self, wrapper_cls, fit_kwargs=None, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        if fit_kwargs is None:
            fit_kwargs = {}
        self.fit_kwargs = fit_kwargs
        self.wrapper_cls = wrapper_cls
        self.estimator = None

    def _build_estimator(self, task, kwargs):
        raise NotImplementedError

    def build_estimator(self, task=None):
        pv = self.param_values
        self.estimator = self.wrapper_cls(self.fit_kwargs, **pv)
        return self.estimator

    def _on_params_ready(self):
        pass

    def _compile(self):
        pass

    def _forward(self, inputs):
        return self.estimator
