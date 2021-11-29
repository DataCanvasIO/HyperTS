from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.datasets import load_arrow_head

from hyperts.estimators import SKTimeWrapper, VARWrapper
from .datasets import *


class Test_Estimator():

    def test_VAR_wrapper(self):
        X, y = get_random_multivariate_forecast_dataset()
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, shuffle=False)
        fit_kwargs = {'timestamp': 'ds'}
        model = VARWrapper(fit_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]

    def test_SKTime_wrapper(self):
        X, y = load_arrow_head(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        fit_kwargs = {}
        classifier = SKTimeWrapper(fit_kwargs=fit_kwargs, n_estimators=200)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        assert score > 0
