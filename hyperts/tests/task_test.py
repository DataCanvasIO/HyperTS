# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sktime.datasets import load_arrow_head

from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hyperts.experiment import TSExperiment
from hyperts.hyper_ts import HyperTS
from hyperts.search_space import search_space_univariate_forecast_generator, search_space_multivariate_forecast_generator, \
    space_classification_classification
from .datasets import *


class Test_Task():

    def test_univariate_forecast(self):
        X, y = get_random_univariate_forecast_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        rs = RandomSearcher(search_space_univariate_forecast_generator(covariate=['id'], time_series='ds'), optimize_direction=OptimizeDirection.Minimize)
        hyper_model = HyperTS(rs, task='univariate-forecast', reward_metric='neg_mean_squared_error', callbacks=[SummaryCallback()])

        exp = TSExperiment(hyper_model, X_train, y_train, time_series_col='ds', X_eval=X_test, y_eval=y_test)
        pipeline_model = exp.run(max_trials=3)

        y_pred = pipeline_model.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]

    def test_multivariate_forecast(self):

        X, y = get_random_multivariate_forecast_dataset()
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, shuffle=False)

        rs = RandomSearcher(search_space_multivariate_forecast_generator(time_series='ds'), optimize_direction=OptimizeDirection.Minimize)
        hyper_model = HyperTS(rs, task='multivariate-forecast', reward_metric='neg_mean_squared_error')

        exp = TSExperiment(hyper_model, X_train, y_train, time_series_col='ds', X_eval=X_test, y_eval=y_test)
        pipeline_model = exp.run(max_trials=3)

        y_pred = pipeline_model.predict(X_test)
        assert y_pred.shape[1] == 2
        assert y_pred.shape[0] == X_test.shape[0]

    def test_univariate_classification(self):
        X, y = load_arrow_head(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        rs = RandomSearcher(space_classification_classification, optimize_direction=OptimizeDirection.Maximize)
        hyper_model = HyperTS(rs, task='binary-classification', reward_metric='accuracy')

        exp = TSExperiment(hyper_model, X_train, y_train, X_eval=X_test, y_eval=y_test)
        pipeline_model = exp.run(max_trials=3)

        y_pred = pipeline_model.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
