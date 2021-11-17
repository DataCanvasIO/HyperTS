# -*- coding:utf-8 -*-
"""

"""
import sys
import lightgbm
import numpy as np
import pandas as pd

from hypernets.core.search_space import ModuleSpace
from hypernets.discriminators import UnPromisingTrial
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.column_selector import column_object_category_bool, column_zero_or_positive_int32
from hypernets.utils import const, logging
from sklearn import metrics as sk_metrics

import time
import pickle

from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.utils import logging, fs
from fbprophet import Prophet

from statsmodels.tsa.vector_ar.var_model import VAR


from hypernets.dispatchers.in_process_dispatcher import InProcessDispatcher

logger = logging.get_logger(__name__)


class EstimatorWrapper:
    def fit(self, X, y):
        pass

    def predict(self, periods):
        pass


class ProphetWrapper(EstimatorWrapper):

    def __init__(self, **kwargs):
        print("args:")
        print(kwargs)
        self.model = Prophet(**kwargs)

    def fit(self, X, y):
        # adapt for prophet
        df_train = X[['ds']]
        df_train['y'] = y
        self.model.fit(df_train)

    def predict(self, X):
        df_predict = self.model.predict(X)
        return df_predict['yhat'].values


class VARWrapper(EstimatorWrapper):

    def __init__(self,  **kwargs):
        if kwargs is None:
            kwargs = {}
        self.init_kwargs = kwargs
        print("VARWrapper.__init__ args:")
        print(self.init_kwargs)
        self.model = None

        # fitted
        self._start_date = None
        self._end_date = None
        self._freq = None
        self._targets = []

    def fit(self, X, y):
        # adapt for prophet
        # init_kwargs
        # 记录模型训练的开始时间和结束时间
        #
        date_series_top2 = X['ds'][:2].tolist()
        self._freq = (date_series_top2[1] - date_series_top2[0]).total_seconds()

        self._start_date = X['ds'].head(1).to_list()[0].to_pydatetime()
        self._end_date = X['ds'].tail(1).to_list()[0].to_pydatetime()

        model = VAR(endog=y, dates=X['ds'])
        self.model = model.fit(**self.init_kwargs)

    def predict(self, X):

        last_date = X['ds'].tail(1).to_list()[0].to_pydatetime()
        steps = int((last_date - self._end_date).total_seconds()/self._freq)
        predict_result = self.model.forecast(self.model.y, steps=steps)

        def calc_index(date):
            r_i = int((date - self._end_date).total_seconds()/self._freq) - 1
            return predict_result[r_i].tolist()

        return np.array(X['ds'].map(calc_index).to_list())


class TSEstimatorMS(ModuleSpace):
    def __init__(self, wrapper_cls, fit_kwargs={}, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.fit_kwargs = fit_kwargs
        self.wrapper_cls = wrapper_cls
        self.estimator = None

    def _build_estimator(self, task, kwargs):
        raise NotImplementedError

    def build_estimator(self, task=None):
        pv = self.param_values
        self.estimator = self.wrapper_cls(**pv)
        return self.estimator

    def _on_params_ready(self):
        pass

    def _compile(self):
        pass

    def _forward(self, inputs):
        return self.estimator


class HyperTSEstimator(Estimator):
    def __init__(self, task, space_sample, data_cleaner_params=None):
        super(HyperTSEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.data_cleaner_params = data_cleaner_params
        self.model = None  # Time-Series model
        self.cv_gbm_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self.pos_label = None
        self.transients_ = {}

        self._build_model(space_sample)

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'

        self.model = outputs[0].build_estimator()

        # logger.debug(f'data_pipeline:{self.data_pipeline}')
        # todo self.pipeline_signature = self.get_pipeline_signature(self.data_pipeline)
        # self.model = ProphetWrapper(**sampled_estimator_params)

    def summary(self):
        s = f"{self.data_pipeline.__repr__(1000000)}"
        return s

    def fit_cross_validation(self, X, y, verbose=0, stratified=True, num_folds=3, pos_label=None,
                             shuffle=False, random_state=9527, metrics=None, **kwargs):
        return None, None, None

    def get_iteration_scores(self):
        return None

    def fit(self, X, y, pos_label=None, verbose=0, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, verbose=0, **kwargs):
        return self.model.predict(X)

    def predict_proba(self, X, verbose=0, **kwargs):
        return None

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        y_pred = self.model.predict(X)
        if self.task == 'multivariate-forecast':
            scores = []
            for i in range(y.shape[1]):
                y_true_part = y[:, i]
                y_pred_part = y_pred[:, i]
                score_part = sk_metrics.mean_squared_error(y_true_part, y_pred_part)  # todo calc mse
                scores.append(score_part)
            score = np.mean(scores)
        else:
            score = sk_metrics.mean_squared_error(y, y_pred)  # todo calc mse
        return {'neg_mean_squared_error': score}

    def save(self, model_file):
        with fs.open(f'{model_file}', 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(model_file):
        with fs.open(f'{model_file}', 'rb') as input:
            model = pickle.load(input)
            return model


class HyperTS(HyperModel):

    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric='accuracy', task=None,
                 discriminator=None, data_cleaner_params=None, cache_dir=None, clear_cache=None):
        self.data_cleaner_params = data_cleaner_params
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric,
                            task=task, discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = HyperTSEstimator(task=self.task, space_sample=space_sample, data_cleaner_params=self.data_cleaner_params)
        return estimator

    def load_estimator(self, model_file):
        return HyperTSEstimator.load(model_file)

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'

    def search(self, X, y, X_eval, y_eval, max_trials=3, **kwargs):
        dispatcher = InProcessDispatcher('/tmp/tmp_data')
        dispatcher.dispatch(self, X, y, X_eval, y_eval, cv=False, num_folds=1, max_trials=max_trials, dataset_id='abc', trial_store=None)
