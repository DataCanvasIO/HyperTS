from hypergbm.cfg import HyperGBMCfg as cfg
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator, HistGBEstimator
from hypergbm.pipeline import DataFrameMapper
from hypergbm.sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, \
    categorical_pipeline_simple, categorical_pipeline_complex, \
    datetime_pipeline_simple, text_pipeline_simple
from hypernets.core import randint
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import HyperSpace, Choice, Int, ModuleSpace
from hypernets.tabular.column_selector import column_object
from hypernets.utils import logging, get_params
from hyperts.hyper_ts import HyperTS, ProphetWrapper
from sklearn.model_selection import train_test_split


from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils

from hyperts.search_space import ts_multivariate_stats_search_space, ts_stats_search_space
from hyperts.experiment import TSExperiment
import pandas as pd

from hyperts.hyper_ts import VARWrapper
from random import random
import datetime


logger = logging.get_logger(__name__)

now_date = datetime.datetime.now()


# contrived dataset with dependency
data = list()
X = []
for i in range(100):
    now_date = now_date + datetime.timedelta(days=1)
    X.append(now_date)
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)

# fit model
X = pd.DataFrame(data={'ds': X})
print(X)

# y_train
y = pd.DataFrame(data=data)
y.columns = ['var_1', 'var_2']
print(y)

X_train, X_test, y_train, y_test, =  train_test_split(X, y, test_size=0.2, shuffle=False)

model = VARWrapper()
model.fit(X, y)


rs = RandomSearcher(ts_multivariate_stats_search_space, optimize_direction=OptimizeDirection.Maximize)
hyper_model = HyperTS(rs, task='multivariate-forecast', reward_metric='neg_mean_squared_error')

exp = TSExperiment(hyper_model, X_train, y_train, 'ds', covariate_cols=None, task='multivariate-forecast', covariate_data_clean_args=None, X_eval=X_test, y_eval=y_test, scorer='neg_mean_squared_error')
pipeline_model = exp.run(max_trials=3)
print(pipeline_model)


# 评估模型指标在外部进行
# result = pipeline_model.evaluate(X_test, y_test, metrics=['neg_mean_squared_error'])
# print(f'final result:{result}')


