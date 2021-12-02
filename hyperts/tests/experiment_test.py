import pandas as pd

from hyperts.utils import consts
from hyperts.utils import data_ops as dp
from hyperts.mk_experiment import make_experiment, test_data_process

class Test_Experiment():

    def test_univariable_forecast(self):
        df = pd.read_csv('../../datasets/network_traffic_forecast.csv', encoding='utf-8')
        df = df[['TimeStamp', 'Var_1', 'HourSin', 'WeekCos', 'CBWD']]

        train_df, test_df = dp.temporal_train_test_split(df, test_size=0.2)

        timestamp = 'TimeStamp'
        covariables = ['HourSin', 'WeekCos', 'CBWD']
        task = consts.TASK_UNIVARIABLE_FORECAST
        optimize_direction = consts.OptimizeDirection_Minimize

        exp = make_experiment(train_df,
                              timestamp=timestamp,
                              covariables=covariables,
                              task=task,
                              callbacks=None,
                              reward_metric='rmse',
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3)

        X_test, y_test = test_data_process(test_df, timestamp, covariables, impute=True)

        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_multivariable_forecast(self):
        df = pd.read_csv('../../datasets/network_traffic_forecast.csv', encoding='utf-8')

        train_df, test_df = dp.temporal_train_test_split(df, test_size=0.2)

        timestamp = 'TimeStamp'
        covariables = ['HourSin', 'WeekCos', 'CBWD']
        task = consts.TASK_MULTIVARIABLE_FORECAST
        optimize_direction = consts.OptimizeDirection_Minimize

        exp = make_experiment(train_df,
                              timestamp=timestamp,
                              covariables=covariables,
                              task=task,
                              callbacks=None,
                              reward_metric='rmse',
                              optimize_direction=optimize_direction)

        model = exp.run(max_trials=3)

        X_test, y_test = test_data_process(test_df, timestamp, covariables, impute=True)

        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape
