# -*- coding:utf-8 -*-
"""

"""
import pickle

import numpy as np
from sklearn import metrics as sk_metrics

from hypernets.dispatchers.in_process_dispatcher import InProcessDispatcher
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.utils import logging, fs

logger = logging.get_logger(__name__)


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
        return "HyperTSEstimator"

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
        if self.task == HyperTS.TASK_MULTIVARIATE_FORECAST:
            scores = []
            for i in range(y.shape[1]):
                y_true_part = y[:, i]
                y_pred_part = y_pred[:, i]
                score_part = sk_metrics.mean_squared_error(y_true_part, y_pred_part)  # todo calc mse
                scores.append(score_part)
            score = np.mean(scores)
            return {'neg_mean_squared_error': score}
        elif self.task == HyperTS.TASK_BINARY_CLASSIFICATION:
            score = sk_metrics.accuracy_score(y, y_pred)
            return {'accuracy': score}
        # TODO: others task types and metrics
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

    TASK_BINARY_CLASSIFICATION = 'binary-classification'
    TASK_MULTIVARIATE_FORECAST = 'multivariate-forecast'
    TASK_UNIVARIATE_FORECAST = 'univariate-forecast'

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

    def search(self, X, y, X_eval, y_eval, max_trials=3, dataset_id=None, trial_store=None, **kwargs):
        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)

        for callback in self.callbacks:
            callback.on_search_start(self, X, y, X_eval, y_eval, None, None, max_trials, dataset_id, trial_store=trial_store)

        dispatcher = InProcessDispatcher('/tmp/tmp_data')  # TODO:
        dispatcher.dispatch(self, X, y, X_eval, y_eval, cv=False, num_folds=None, max_trials=max_trials, dataset_id=dataset_id, trial_store=trial_store)
