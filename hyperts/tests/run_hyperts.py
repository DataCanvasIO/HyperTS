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
from hyperts.hyper_ts import HyperTS, ProphetWrapper, ProphetEstimatorMS
from sklearn.model_selection import train_test_split


from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils


logger = logging.get_logger(__name__)


# todo define a search space
def search_space_one_trial():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        ProphetEstimatorMS(interval_width=Choice([0.5, 0.6, 0.7, 0.8]), seasonality_mode=Choice(['additive', 'multiplicative']))(input)
        space.set_inputs(input)
    return space


rs = RandomSearcher(search_space_one_trial, optimize_direction=OptimizeDirection.Maximize)
hk = HyperTS(rs, reward_metric='neg_mean_squared_error')

target = 'y'
df = pd.read_csv('C:/Users/wuhf/OpenSource/prophet/examples/example_wp_log_peyton_manning.csv')
y = df.pop(target)
X = df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = hk.search(X_train, y_train, X_test, y_test, max_trials=3)
best_trial = hk.get_best_trial()
print(f'best_train:{best_trial}')
print(model)



# result = estimator.evaluate(X_test, y_test, metrics=['auc', 'accuracy'])
# print(f'final result:{result}')





