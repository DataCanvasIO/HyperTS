from sktime.datasets import load_arrow_head

import hyperts.utils.toolbox as dp
from hyperts.utils.metrics import accuracy_score
from hyperts.framework.wrappers.stats_wrappers import ProphetWrapper, ARIMAWrapper, VARWrapper, TSFClassifierWrapper
from hyperts.datasets import load_random_univariate_forecast_dataset, load_random_multivariate_forecast_dataset


class Test_Estimator():

    def test_Prophet_wrapper(self):
        X, y = load_random_univariate_forecast_dataset()
        X_train, X_test, y_train, y_test, = dp.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        model = ProphetWrapper(fit_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_ARIMA_wrapper(self):
        X, y = load_random_univariate_forecast_dataset()
        X_train, X_test, y_train, y_test, = dp.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        model = ARIMAWrapper(fit_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_VAR_wrapper(self):
        X, y = load_random_multivariate_forecast_dataset()
        X_train, X_test, y_train, y_test, = dp.temporal_train_test_split(X, y, test_size=0.2)
        fit_kwargs = {'timestamp': 'ds'}
        model = VARWrapper(fit_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_TSF_wrapper(self):
        X, y = load_arrow_head(return_X_y=True)
        X_train, X_test, y_train, y_test = dp.random_train_test_split(X, y)
        classifier = TSFClassifierWrapper(fit_kwargs=None, n_estimators=200)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        assert score > 0
