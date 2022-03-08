# -*- coding:utf-8 -*-
"""

"""
import copy

import time
import pickle

import numpy as np
from sklearn import pipeline as sk_pipeline

from hypernets.utils import fs, logging
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.core.meta_learner import MetaLearner
from hypernets.pipeline.base import ComposeTransformer
from hypernets.dispatchers.in_process_dispatcher import InProcessDispatcher

from hyperts.utils import consts, get_tool_box


logger = logging.get_logger(__name__)


class HyperTSEstimator(Estimator):
    """A `Estimator` object about Time Series.

    Parameters
    ----------
    task: 'str'.
        Task could be 'univariate-forecast', 'multivariate-binaryclass', etc.
        See consts.py for details.
    mode: 'str'.
        The hyperts can support three mode: 'dl', 'stats', and 'nas'.
    space_sample: An instance class representing a hyperts estimator.
    data_cleaner_params: 'dirt' or None, default None.
        For details of parameters, refer to hypernets.tabular.data_cleaner.
    """

    def __init__(self, task, mode, space_sample, data_cleaner_params=None):
        super(HyperTSEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.mode = mode
        self.data_cleaner_params = data_cleaner_params
        self.model = None  # Time-Series model
        self.cv_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self.pos_label = None
        self.history_prior = None
        self.transients_ = {}

        self._build_model(space_sample)

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.model = outputs[0].estimator
        self.fit_kwargs = outputs[0].fit_kwargs

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        if isinstance(pipeline_module[0], ComposeTransformer):
            self.data_pipeline = self.build_pipeline(space, pipeline_module[0])

    def build_pipeline(self, space, last_transformer):
        transformers = []
        while True:
            next, (name, p) = last_transformer.compose()
            transformers.insert(0, (name, p))
            inputs = space.get_inputs(next)
            if inputs == space.get_inputs():
                break
            assert len(inputs) == 1, 'The `ComposeTransformer` can only contains 1 input.'
            assert isinstance(inputs[0], ComposeTransformer), \
                'The upstream node of `ComposeTransformer` must be `ComposeTransformer`.'
            last_transformer = inputs[0]
        assert len(transformers) > 0
        if len(transformers) == 1:
            return transformers[0][1]
        else:
            pipeline = sk_pipeline.Pipeline(steps=transformers)
            return pipeline

    def summary(self):
        if self.data_pipeline is not None:
            return f"{self.data_pipeline.__repr__(1000000)}"
        else:
            return "HyperTSEstimator"

    def fit_cross_validation(self, X, y, verbose=0, stratified=True, num_folds=3, pos_label=None,
                             shuffle=False, random_state=9527, metrics=None, **kwargs):
        return None, None, None

    def get_iteration_scores(self):
        return None

    def fit(self, X, y, pos_label=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')
        self.pos_label = pos_label

        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.fit_transform(X)
        else:
            X_transformed = X
        self.model.fit(X_transformed, y, **kwargs)

        if self.task in consts.TASK_LIST_FORECAST:
            tb = get_tool_box(y)
            self.history_prior = tb.df_mean_std(y)

        if self.classes_ is None and hasattr(self.model, 'classes_'):
            self.classes_ = self.model.classes_

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

    def predict(self, X, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is predicting the data')

        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.transform(X)
        else:
            X_transformed = X

        if self.cv_models_ is not None:
            raise NotImplementedError('The current version does not support CV.')
        else:
            preds = self.model.predict(X_transformed, **kwargs)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return preds

    def predict_proba(self, X, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is predicting the data')

        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.transform(X)
        else:
            X_transformed = X

        if hasattr(self.model, 'predict_proba'):
            method = 'predict_proba'
        else:
            method = 'predict'

        if self.cv_models_ is not None:
            raise  NotImplementedError('The current version does not support CV.')
        else:
            proba = getattr(self.model, method)(X_transformed, **kwargs)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return proba

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if metrics is None:
            if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
                metrics = ['rmse']
            elif self.task in consts.TASK_LIST_CLASSIFICATION:
                metrics = ['accuracy']

        y_pred = self.predict(X, verbose=verbose)

        if self.task in consts.TASK_LIST_CLASSIFICATION:
            y_proba = self.predict_proba(X, verbose=verbose)
            if 'binaryclass' in self.task:
                classification_type = 'binary'
            else:
                classification_type = 'multiclass'
            scores = get_tool_box(X).metrics.calc_score(y, y_pred, y_proba,
                                                        metrics=metrics,
                                                        task=classification_type,
                                                        pos_label=self.pos_label,
                                                        classes=self.classes_)
        else:
            scores = get_tool_box(X).metrics.calc_score(y, y_pred,
                                                        metrics=metrics,
                                                        task=self.task)

        return scores

    def save(self, model_file):
        if self.mode == consts.Mode_STATS:
            with fs.open(f'{model_file}', 'wb') as output:
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            subself = copy.copy(self)
            subself.model.model.save_model(model_file)
            with fs.open(f'{model_file}', 'wb') as output:
                pickle.dump(subself, output, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(model_file, mode):
        if mode == consts.Mode_STATS:
            with fs.open(f'{model_file}', 'rb') as input:
                estimator = pickle.load(input)
        else:
            from hyperts.framework.dl.models import BaseDeepEstimator
            with fs.open(f'{model_file}', 'rb') as input:
                estimator = pickle.load(input)
            model = BaseDeepEstimator.load_model(model_file)
            estimator.model.model.model = model
        return estimator


class HyperTS(HyperModel):
    """A `HyperModel` object about Time Series.

    Parameters
    ----------
    searcher: 'str', searcher class, search object.
        Searchers come from hypernets, such as EvolutionSearcher, MCTSSearcher, or RandomSearcher, etc.
        See hypernets.searchers for details.
    task: 'str' or None, default None.
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
        See consts.py for details.
    mode: 'str', default 'stats'.
        The hyperts can support three mode: 'dl', 'stats', and 'nas'.
    dispatcher: class object or None, default None.
        Dispatcher is used to provide different execution modes for search trials,
        such as in process mode (`InProcessDispatcher`), distributed parallel mode (`DaskDispatcher`), etc.
    callbacks: list of ExperimentCallback or None, default None.
    reward_metric: 'str' or callable.
        Default 'accuracy' for binary/multiclass task, 'rmse' for forecast/regression task.
    discriminator: Instance of hypernets.discriminator.BaseDiscriminator, which is used to determine
        whether to continue training. Default None.
    data_cleaner_params: 'dirt' or None, default None.
        For details of parameters, refer to hypernets.tabular.data_cleaner.
    clear_cache: 'bool', default False.
        Clear cache store before running the expeirment.

    Returns
    -------
    hyper_ts_cls: subclass of HyperModel
        Subclass of HyperModel to run trials within the experiment.
    """

    def __init__(self,
                 searcher,
                 task=None,
                 mode='stats',
                 dispatcher=None,
                 callbacks=None,
                 reward_metric='accuracy',
                 discriminator=None,
                 data_cleaner_params=None,
                 clear_cache=False):

        self.mode = mode
        self.data_cleaner_params = data_cleaner_params

        HyperModel.__init__(self,
                            searcher,
                            dispatcher=dispatcher,
                            callbacks=callbacks,
                            reward_metric=reward_metric,
                            task=task,
                            discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = HyperTSEstimator(task=self.task, mode=self.mode, space_sample=space_sample, data_cleaner_params=self.data_cleaner_params)
        return estimator

    def load_estimator(self, model_file):
        return HyperTSEstimator._load(model_file, self.mode)

    def _get_reward(self, value, key=None):
        def cast_float(value):
            try:
                fv = float(value)
                return fv
            except TypeError:
                return None

        if isinstance(value, dict) and isinstance(key, str):
            reward = cast_float(value[key])
        elif isinstance(value, dict) and not isinstance(key, str):
            reward = cast_float(value[key.__name__])
        else:
            raise ValueError(f'"{key}" should be a string or function name for metric.')

        return reward

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'

    def search(self, X, y, X_eval, y_eval, cv=False, num_folds=3, max_trials=3, dataset_id=None, trial_store=None, **fit_kwargs):
        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)

        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trial_store))

        self._before_search()

        # dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)
        dispatcher = InProcessDispatcher('/models')   # TODO：

        for callback in self.callbacks:
            callback.on_search_start(self, X, y, X_eval, y_eval,
                                     cv, num_folds, max_trials, dataset_id, trial_store,
                                     **fit_kwargs)
        try:
            trial_no = dispatcher.dispatch(self, X, y, X_eval, y_eval,
                                           cv, num_folds, max_trials, dataset_id, trial_store,
                                           **fit_kwargs)

            for callback in self.callbacks:
                callback.on_search_end(self)
        except Exception as e:
            for callback in self.callbacks:
                callback.on_search_error(self)
            raise e

        self._after_search(trial_no)