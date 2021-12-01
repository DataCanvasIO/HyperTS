import pandas as pd

from hypernets.core.searcher import OptimizeDirection

from hyperts.utils import consts
from hyperts.utils import data_ops as dp
from hyperts.mk_experiment import make_experiment

if __name__ == '__main__':
    df = pd.read_csv('../../datasets/network_traffic_forecast.csv', encoding='utf-8')

    train_df, test_df = dp.temporal_train_test_split(df, test_size=0.2)

    task = consts.TASK_MULTIVARIABLE_FORECAST
    timestamp = 'TimeStamp'
    covariables = ['HourSin', 'WeekCos', 'CBWD']

    exp = make_experiment(train_df,
                          timestamp=timestamp,
                          covariables=covariables,
                          task=task,
                          callbacks=None,
                          reward_metric='rmse',
                          optimize_direction=OptimizeDirection.Minimize)

    model = exp.run(max_trials=3)