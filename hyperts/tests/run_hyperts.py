import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from hypernets.core.callbacks import *
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.utils import fs
from hyperts.hyper_ts import HyperTS
from hyperts.estimators import ProphetWrapper, TSEstimatorMS

logger = logging.get_logger(__name__)


def search_space_one_trial():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(ProphetWrapper, interval_width=Choice([0.5, 0.6]), seasonality_mode=Choice(['additive', 'multiplicative']))(input)
        space.set_inputs(input)
    return space


rs = RandomSearcher(search_space_one_trial, optimize_direction=OptimizeDirection.Maximize)
ht = HyperTS(rs, reward_metric='neg_mean_squared_error')


X = pd.DataFrame({'ds': pd.date_range("2013-01-01", periods=100, freq='D')})
y = pd.DataFrame({'value':  np.random.rand(1, 100)[0].tolist()})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

ht.search(X_train, y_train, X_test, y_test, max_trials=1)
best_trial = ht.get_best_trial()
print(f'best_train:{best_trial}')
assert best_trial

with fs.open(best_trial.model_file, 'rb') as f:
    import pickle as pkl
    estimator = pkl.load(f)
result = estimator.evaluate(X_test, y_test, metrics=['auc', 'accuracy'])
print(f'final result:{result}')

assert result




