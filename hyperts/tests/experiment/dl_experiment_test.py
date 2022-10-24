import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd

from hyperts.datasets import load_network_traffic, \
                             load_arrow_head, \
                             load_basic_motions, \
                             load_real_known_cause_dataset
from hyperts.utils import consts, metrics
from hyperts.utils._base import get_tool_box
from hyperts.experiment import make_experiment
from hyperts.tests import skip_if_not_tf


@skip_if_not_tf
class Test_Experiment():

    def test_univariate_forecast(self):
        df = load_network_traffic(univariate=True)
        tb = get_tool_box(df)
        train_df, test_df = tb.temporal_train_test_split(df, test_size=0.1)

        timestamp = 'TimeStamp'
        covariables = ['HourSin', 'WeekCos', 'CBWD']
        task = consts.Task_FORECAST
        reward_metric = metrics.smape
        optimize_direction = consts.OptimizeDirection_MINIMIZE

        exp = make_experiment(train_df.copy(),
                              mode='dl',
                              timestamp=timestamp,
                              covariables=covariables,
                              task=task,
                              cv=False,
                              reward_metric=reward_metric,
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3, epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_df.copy())

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]
        score = model.evaluate(y_test, y_pred)
        print('univariate_forecast score: ', score)

    def test_multivariate_forecast(self):
        df = load_network_traffic()
        tb = get_tool_box(df)
        train_df, test_df = tb.temporal_train_test_split(df, test_size=0.1)

        timestamp = 'TimeStamp'
        covariables = ['HourSin', 'WeekCos', 'CBWD']
        task = consts.Task_MULTIVARIATE_FORECAST
        reward_metric = consts.Metric_RMSE
        optimize_direction = consts.OptimizeDirection_MINIMIZE

        exp = make_experiment(train_df.copy(),
                              mode='dl',
                              timestamp=timestamp,
                              covariables=covariables,
                              task=task,
                              cv=False,
                              reward_metric=reward_metric,
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3, epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_df.copy())

        y_pred = model.predict(X_test)
        assert y_pred.shape[0] == y_test.shape[0]
        score = model.evaluate(y_test, y_pred)
        print('multivariate_forecast score: ', score)

    def test_univariate_classification(self):
        df = load_arrow_head()
        tb = get_tool_box(df)
        train_df, test_df = tb.random_train_test_split(df, test_size=0.2, random_state=9527)

        target = 'target'
        task = consts.Task_CLASSIFICATION
        reward_metric = consts.Metric_ACCURACY
        optimize_direction = consts.OptimizeDirection_MAXIMIZE

        exp = make_experiment(train_df.copy(),
                              mode='dl',
                              task=task,
                              cv=False,
                              eval_data=test_df.copy(),
                              target=target,
                              reward_metric=reward_metric,
                              random_state=9527,
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3, epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_df.copy())

        y_pred = model.predict(X_test)

        score = metrics.accuracy_score(y_test, y_pred)

        assert score > 0
        print('univariate_classification accuracy:  {} %'.format(score*100))

    def test_multivariate_classification(self):
        df = load_basic_motions()
        tb = get_tool_box(df)
        train_df, test_df = tb.random_train_test_split(df, test_size=0.2, random_state=9527)

        target = 'target'
        task = consts.Task_CLASSIFICATION
        reward_metric = metrics.f1_score
        optimize_direction = consts.OptimizeDirection_MAXIMIZE

        exp = make_experiment(train_df.copy(),
                              mode='dl',
                              task=task,
                              cv=False,
                              eval_data=test_df.copy(),
                              target=target,
                              reward_metric=reward_metric,
                              random_state=9527,
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3, epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_df.copy())

        y_pred = model.predict(X_test)

        score = metrics.accuracy_score(y_test, y_pred)

        assert score > 0
        print('multivariate_classification accuracy:  {} %'.format(score*100))


    def test_univariate_anomaly_detection(self):
        df = load_real_known_cause_dataset()
        df_ = df.drop(columns=['anomaly'])
        tb = get_tool_box(df)
        train_df, test_df = tb.temporal_train_test_split(df_, test_horizon=15000)

        exp = make_experiment(train_data=train_df.copy(),
                              mode='dl',
                              task='detection',
                              timestamp='timestamp',
                              cv=False,
                              ensemble_size=None,
                              random_state=2022)

        model = exp.run(max_trials=3, epochs=10, final_train_epochs=10)

        X_test, _ = model.split_X_y(test_df.copy())
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        y_test = df.iloc[-15000:, 2]

        scores = model.evaluate(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

        assert isinstance(scores, pd.DataFrame)
        assert y_pred.shape[0] == y_test.shape[0]
        print('univariate_anomaly_detection score: ', scores)