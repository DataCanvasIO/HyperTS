import numpy as np
from hyperts.utils._base import get_tool_box
from hyperts.utils.metrics import accuracy_score
from hyperts.framework.wrappers.stats_wrappers import ProphetWrapper
from hyperts.framework.wrappers.stats_wrappers import ARIMAWrapper
from hyperts.framework.wrappers.stats_wrappers import VARWrapper
from hyperts.framework.wrappers.stats_wrappers import TSForestWrapper
from hyperts.framework.wrappers.stats_wrappers import KNeighborsWrapper
from hyperts.framework.wrappers.stats_wrappers import IForestWrapper
from hyperts.framework.wrappers.stats_wrappers import OneClassSVMWrapper
from hyperts.datasets import load_random_univariate_forecast_dataset
from hyperts.datasets import load_random_multivariate_forecast_dataset
from hyperts.datasets import load_arrow_head
from hyperts.datasets import load_real_known_cause_dataset
from hyperts.tests import skip_if_not_prophet

class Test_Stats_Wrappers():

    @skip_if_not_prophet
    def test_Prophet_wrapper(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test, = tb.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        init_kwargs = {
            'seasonality_mode': 'multiplicative'
        }
        model = ProphetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_ARIMA_wrapper(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test, = tb.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        init_kwargs = {
            'p': 1,
            'd': 1,
            'q': 2,
            'trend': 'c',
            'y_scale': np.random.choice(['min_max', 'max_abs', 'none'], size=1)[0]

        }
        model = ARIMAWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_VAR_wrapper(self):
        X, y = load_random_multivariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test, = tb.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        init_kwargs = {
            'trend': 'c',
            'y_scale': np.random.choice(['min_max', 'max_abs', 'none'], size=1)[0]
        }
        model = VARWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_TSF_wrapper(self):
        X, y = load_arrow_head(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y)
        fit_kwargs = {}
        init_kwargs = {
            'min_interval': 3,
            'n_estimators': 50,
        }
        model = TSForestWrapper(fit_kwargs=fit_kwargs, **init_kwargs)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        assert score >= 0

    def test_KNeighbors_wrapper(self):
        X, y = load_arrow_head(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y)
        fit_kwargs = {}
        init_kwargs = {
            'n_neighbors': 3,
            'distance': 'ddtw',
            'x_scale': np.random.choice(['min_max', 'z_score'], size=1)[0]
        }
        model = KNeighborsWrapper(fit_kwargs=fit_kwargs, **init_kwargs)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        assert score >= 0

    def test_IForest_wrapper(self):
        X, y = load_real_known_cause_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_horizon=15000)

        fit_kwargs = {
            'timestamp': 'timestamp'
        }
        init_kwargs = {
            'n_estimators': 50,
            'contamination': 0.1,
            'x_scale': np.random.choice(['min_max', 'z_score'], size=1)[0]
        }

        model = IForestWrapper(fit_kwargs=fit_kwargs, **init_kwargs)

        model.fit(X_train)
        y_pred, y_confidence = model.predict(X_test, return_confidence=True)
        score = accuracy_score(y_test, y_pred)
        assert score >= 0

    def test_OneClassSVM_wrapper(self):
        X, y = load_real_known_cause_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_horizon=15000)

        fit_kwargs = {
            'timestamp': 'timestamp'
        }
        init_kwargs = {
            'kernel': 'linear',
            'contamination': 0.1,
            'x_scale': np.random.choice(['min_max', 'z_score'], size=1)[0]
        }

        model = OneClassSVMWrapper(fit_kwargs=fit_kwargs, **init_kwargs)

        model.fit(X_train)
        y_pred, y_confidence = model.predict(X_test, return_confidence=True)
        score = accuracy_score(y_test, y_pred)
        assert score >= 0