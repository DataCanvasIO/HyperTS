import numpy as np

try:
    try:
        from prophet import Prophet
    except:
        from fbprophet import Prophet
    is_prophet_installed = True
except:
    is_prophet_installed = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from hypernets.utils import logging

from hyperts.framework.wrappers import EstimatorWrapper, WrapperMixin, suppress_stdout_stderr

logger = logging.get_logger(__name__)


##################################### Define Time Series Forecast Wrapper #####################################
class ProphetWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(ProphetWrapper, self).__init__(fit_kwargs, **kwargs)
        if is_prophet_installed:
            self.model = Prophet(**self.init_kwargs)
        else:
            self.model = None

    def fit(self, X, y=None, **kwargs):
        # adapt for prophet
        df_train = X[[self.timestamp]]
        if self.timestamp != 'ds':
            df_train.rename(columns={self.timestamp: 'ds'}, inplace=True)
        df_train['y'] = y
        with suppress_stdout_stderr():
            self.model.fit(df_train)

    def predict(self, X, **kwargs):
        df_test = X[[self.timestamp]]
        if self.timestamp != 'ds':
            df_test.rename(columns={self.timestamp: 'ds'}, inplace=True)
        df_preds = self.model.predict(df_test)
        preds = df_preds['yhat'].values.reshape((-1, 1))
        preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
        return preds


class ARIMAWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(ARIMAWrapper, self).__init__(fit_kwargs, **kwargs)
        # fitted
        self.model = None
        self._end_date = None
        self._freq = None

    def fit(self, X, y=None, **kwargs):
        # adapt for prophet
        date_series_top2 = X[self.timestamp][:2].tolist()
        self._freq = (date_series_top2[1] - date_series_top2[0]).total_seconds()
        self._end_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()

        y = self.fit_transform(y)

        p = self.init_kwargs.pop('p', 1)
        d = self.init_kwargs.pop('d', 1)
        q = self.init_kwargs.pop('q', 1)
        trend = self.init_kwargs.pop('trend', 'c')
        seasonal_order = self.init_kwargs.pop('seasonal_order', (0, 0, 0, 0))

        model = ARIMA(endog=y, order=(p, d, q), trend=trend,
                      seasonal_order=seasonal_order, dates=X[self.timestamp])
        self.model = model.fit(**self.init_kwargs)

    def predict(self, X, **kwargs):
        last_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()
        if last_date == self._end_date:
            raise ValueError('The end date of the valid set must be '
                             'less than the end date of the test set.')
        steps = int((last_date - self._end_date).total_seconds() / self._freq)
        predict_result = self.model.forecast(steps=steps).values

        def calc_index(date):
            r_i = int((date - self._end_date).total_seconds() / self._freq) - 1
            return predict_result[r_i].tolist()

        preds = np.array(X[self.timestamp].map(calc_index).to_list()).reshape(-1, 1)
        preds = self.inverse_transform(preds)
        preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
        return preds


class VARWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: multivariate forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(VARWrapper, self).__init__(fit_kwargs, **kwargs)
        # fitted
        self.model = None
        self._end_date = None
        self._freq = None

    def fit(self, X, y=None, **kwargs):
        # adapt for prophet
        date_series_top2 = X[self.timestamp][:2].tolist()
        self._freq = (date_series_top2[1] - date_series_top2[0]).total_seconds()
        self._end_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()

        y = self.fit_transform(y)

        model = VAR(endog=y, dates=X[self.timestamp])
        self.model = model.fit(**self.init_kwargs)

    def predict(self, X, **kwargs):
        last_date = X[self.timestamp].tail(1).to_list()[0].to_pydatetime()
        if last_date == self._end_date:
            raise ValueError('The end date of the valid set must be '
                             'less than the end date of the test set.')
        steps = int((last_date - self._end_date).total_seconds() / self._freq)
        predict_result = self.model.forecast(self.model.y, steps=steps)

        def calc_index(date):
            r_i = int((date - self._end_date).total_seconds() / self._freq) - 1
            return predict_result[r_i].tolist()

        preds = np.array(X[self.timestamp].map(calc_index).to_list())
        preds = self.inverse_transform(preds)
        preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
        return preds


##################################### Define Time Series Classification Wrapper #####################################
class TSForestWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate classification.
    """
    def __init__(self, fit_kwargs=None, **kwargs):
        super(TSForestWrapper, self).__init__(fit_kwargs, **kwargs)
        self.model = TimeSeriesForestClassifier(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        # adapt for prophet
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_


class KNeighborsWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate/multivariate classification.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(KNeighborsWrapper, self).__init__(fit_kwargs, **kwargs)
        self.model = KNeighborsTimeSeriesClassifier(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        # adapt for prophet
        X = self.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        X = self.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        X = self.transform(X)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_