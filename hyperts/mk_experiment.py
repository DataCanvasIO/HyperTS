# -*- coding:utf-8 -*-
"""

"""
from hypernets.core.callbacks import SummaryCallback
from hypernets.searchers.random_searcher import RandomSearcher

from hyperts.utils import consts
from hyperts.utils import data_ops as dp
from hyperts.hyper_ts import HyperTS
from hyperts.experiment import TSExperiment
from hyperts.macro_search_space import stats_forecast_search_space, stats_classification_search_space


def make_experiment(train_df,
                    eval_df=None,
                    timestamp=None,
                    covariables=None,
                    task=None,
                    callbacks=None,
                    searcher=None,
                    reward_metric=None,
                    optimize_direction=None,
                    **kwargs):

    target_varibales = dp.list_diff(train_df.columns.tolist(), [timestamp]+covariables)
    X_train = train_df[[timestamp]+covariables]
    y_train = train_df[target_varibales]

    if eval_df is not None:
        X_eval, y_eval = eval_df[[timestamp]+covariables], eval_df[target_varibales]
    else:
        X_eval, y_eval = None, None

    if task in [consts.TASK_FORECAST, consts.TASK_UNIVARIABLE_FORECAST, consts.TASK_MULTIVARIABLE_FORECAST]:
        if len(y_train.columns) == 1:
            task = consts.TASK_UNIVARIABLE_FORECAST
        search_pace = stats_forecast_search_space(task=task, timestamp=timestamp, covariables=covariables)
    else:
        search_pace = stats_classification_search_space(task=task, timestamp=timestamp)

    if searcher ==None:
        searcher = RandomSearcher(search_pace, optimize_direction=optimize_direction)
    hyper_model = HyperTS(searcher, reward_metric=reward_metric, task=task, callbacks=[SummaryCallback()])

    experiment = TSExperiment(hyper_model, X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                timestamp_col=timestamp, covariate_cols=covariables, task=task)

    return experiment