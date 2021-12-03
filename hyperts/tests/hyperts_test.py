from hypernets.core.ops import HyperInput
from hypernets.core.callbacks import SummaryCallback
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.searchers.random_searcher import RandomSearcher

from hyperts.hyper_ts import HyperTS
from hyperts.utils import consts, toolbox as dp
from hyperts.datasets import load_random_univariate_forecast_dataset
from hyperts.framework.wrappers.stats_wrappers import SimpleTSEstimator, ProphetWrapper

class Test_HyperTS():

    @classmethod
    def search_space_one_trial(cls, timestamp):
        fit_kwargs = {'timestamp': timestamp}
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            SimpleTSEstimator(ProphetWrapper, fit_kwargs=fit_kwargs, seasonality_mode=Choice(['additive', 'multiplicative']))(input)
            space.set_inputs(input)
        return space

    def test_hyperts(self):

        X, y = load_random_univariate_forecast_dataset()
        X_train, X_test, y_train, y_test = dp.temporal_train_test_split(X, y, test_horizion=24)

        task = consts.Task_UNIVARIABLE_FORECAST
        optimize_direction = consts.OptimizeDirection_MINIMIZE
        reward_metric = consts.Metric_RMSE

        rs = RandomSearcher(lambda : self.search_space_one_trial(timestamp='ds'), optimize_direction=optimize_direction)
        ht = HyperTS(rs, reward_metric=reward_metric, task=task, callbacks=[SummaryCallback()])

        ht.search(X_train, y_train, X_test, y_test, max_trials=1)
        best_trial = ht.get_best_trial()

        estimator = ht.final_train(best_trial.space_sample, X_train, y_train)
        result = estimator.evaluate(X_test, y_test)
        assert result[reward_metric] > 0





