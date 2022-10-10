# -*- coding:utf-8 -*-
"""

"""
import os
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
from hyperts.utils.transformers import IdentityTransformer

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

    def __init__(self, task, mode, reward_metric, space_sample,
                       timestamp=None, covariates=None,
                       data_cleaner_params=None, weights_cache=None):
        super(HyperTSEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.mode = mode
        self.reward_metric = reward_metric
        self.timestamp = timestamp
        self.covariates = covariates
        self.data_cleaner_params = data_cleaner_params
        self.weights_cache = weights_cache
        self.model = None
        self.cv_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self.pos_label = None
        self.transients_ = {}

        self._build_model(space_sample)

    def _build_model(self, space_sample):
        if self.mode != consts.Mode_NAS:
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
        else:
            from hyperts.framework.wrappers.nas_wrappers import TSNASWrapper
            space_sample.weights_cache = self.weights_cache
            init_kwargs = space_sample.__dict__.get('hyperparams').param_values
            self.model = TSNASWrapper(dict(), **init_kwargs)
            self.model.model.space_sample = copy.deepcopy(space_sample)
            self.data_pipeline = IdentityTransformer()

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
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('transforming the train set')

        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        pbar = self.transients_.get('pbar')
        if pbar is not None:
            pbar.reset()
            pbar.set_description('fit_transform_data')

        tb = get_tool_box(X, y)
        if self.data_pipeline is not None:
            X_transformed = self.fit_transform_X(X)
        else:
            X_transformed = X

        cross_validator = kwargs.pop('cross_validator', None)
        if cross_validator is not None:
            iterators = cross_validator
        else:
            if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
                iterators = tb.preqfold(strategy='preq-bls', n_splits=num_folds)
            elif stratified and self.task not in consts.TASK_LIST_REGRESSION:
                iterators = tb.statified_kfold(n_splits=num_folds, shuffle=True, random_state=random_state)
            else:
                iterators = tb.kfold(n_splits=num_folds, shuffle=True, random_state=random_state)

        if metrics is None:
            if self.reward_metric is not None:
                metrics = [self.reward_metric]
            else:
                if self.task in consts.TASK_LIST_FORECAST:
                    metrics = ['mae']
                elif self.task in consts.TASK_LIST_CLASSIFICATION:
                    metrics = ['accuracy']
                elif self.task in consts.TASK_LIST_REGRESSION:
                    metrics = ['rmse']
                elif self.task in consts.TASK_LIST_DETECTION:
                    metrics = ['f1']
                else:
                    raise ValueError(f'This task type [{self.task}] is not supported.')

        oof_ = []
        oof_scores = []
        self.cv_models_ = []
        if pbar is not None:
            pbar.set_description('cross_validation')
        sel = tb.select_1d
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            x_train_fold, y_train_fold = sel(X_transformed, train_idx), sel(y, train_idx)
            x_val_fold, y_val_fold = sel(X_transformed, valid_idx), sel(y, valid_idx)

            fold_est = copy.deepcopy(self.model)
            fold_est.group_id = f'{fold_est.__class__.__name__}_cv_{n_fold}'

            fold_start_at = time.time()
            fold_est.fit(x_train_fold, y_train_fold, **kwargs)
            if verbose:
                logger.info(f'fit fold {n_fold} with {time.time() - fold_start_at} seconds')

            if self.classes_ is None and hasattr(fold_est, 'classes_'):
                self.classes_ = np.array(tb.to_local(fold_est.classes_)[0])

            if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_DETECTION:
                proba = fold_est.predict_proba(x_val_fold)
            else:
                proba = fold_est.predict(x_val_fold)

            fold_scores = self.get_scores(y_val_fold, proba, metrics)
            oof_scores.append(fold_scores)
            oof_.append((valid_idx, proba))
            self.cv_models_.append(fold_est)

            if pbar is not None:
                pbar.update(1)

        logger.info(f'oof_scores: {oof_scores}')
        oof_ = tb.merge_oof(oof_)
        scores = self.get_scores(y, oof_, metrics)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return scores, oof_, oof_scores

    def get_scores(self, y, oof_, metrics):
        tb = get_tool_box(y)
        y, proba = tb.select_valid_oof(y, oof_)
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_DETECTION:
            if 'binaryclass' in self.task or self.task in consts.TASK_LIST_DETECTION:
                classification_type = 'binary'
            else:
                classification_type = 'multiclass'
            preds = self.proba2predict(proba)
            preds = tb.take_array(self.classes_, preds, axis=0)
            scores = tb.metrics.calc_score(y, preds, proba,
                                           metrics=metrics,
                                           task=classification_type,
                                           pos_label=self.pos_label,
                                           classes=self.classes_)
        else:
            scores = tb.metrics.calc_score(y, proba, metrics=metrics, task=self.task)
        return scores

    def get_iteration_scores(self):
        iteration_scores = {}

        def get_scores(ts_model, iteration_scores, fold=None, ):
            if hasattr(ts_model, 'iteration_scores'):
                if ts_model.__dict__.get('group_id'):
                    group_id = ts_model.group_id
                else:
                    if fold is not None:
                        group_id = f'{ts_model.__class__.__name__}_cv_{i}'
                    else:
                        group_id = ts_model.__class__.__name__
                iteration_scores[group_id] = ts_model.iteration_scores

        if self.cv_models_:
            for i, ts_model in enumerate(self.cv_models_):
                get_scores(ts_model, iteration_scores, i)
        else:
            get_scores(self.model, iteration_scores)
        return iteration_scores

    def fit(self, X, y, pos_label=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')

        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        self.pos_label = pos_label

        if self.data_pipeline is not None:
            X_transformed = self.fit_transform_X(X)
        else:
            X_transformed = X
        self.model.fit(X_transformed, y, **kwargs)

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

        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        if self.data_pipeline is not None:
            X_transformed = self.fit_transform_X(X)
        else:
            X_transformed = X

        if self.cv_models_ is not None:
            if self.task in consts.TASK_LIST_REGRESSION:
                pred_sum = None
                for est in self.cv_models_:
                    pred = est.predict(X_transformed)
                    if pred_sum is None:
                        pred_sum = pred
                    else:
                        pred_sum += pred
                preds = pred_sum / len(self.cv_models_)
            else:
                proba = self.predict_proba(X_transformed)
                preds = self.proba2predict(proba)
                preds = get_tool_box(preds).take_array(np.array(self.classes_), preds, axis=0)
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
            X_transformed = self.fit_transform_X(X)
        else:
            X_transformed = X

        if hasattr(self.model, 'predict_proba'):
            method = 'predict_proba'
        else:
            method = 'predict'

        if self.cv_models_ is not None:
            prpba_sum = None
            for est in self.cv_models_:
                proba = getattr(est, method)(X_transformed, **kwargs)
                if prpba_sum is None:
                    prpba_sum = proba
                else:
                    prpba_sum += proba
            proba = prpba_sum / len(self.cv_models_)
        else:
            proba = getattr(self.model, method)(X_transformed, **kwargs)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return proba

    def fit_transform_X(self, X):
        assert self.data_pipeline is not None
        tb = get_tool_box(X)
        if self.timestamp is not None:
            if self.covariates is not None:
                col_transformed = [self.timestamp] + self.covariates
            else:
                col_transformed = [self.timestamp]
            X_transformed = self.data_pipeline.fit_transform(X[col_transformed])
            excluded_variables = tb.list_diff(X.columns.tolist(), col_transformed)
            X_transformed[excluded_variables] = X[excluded_variables]
        else:
            X_transformed = self.data_pipeline.fit_transform(X)

        return X_transformed

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if metrics is None:
            if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
                metrics = ['rmse']
            elif self.task in consts.TASK_LIST_CLASSIFICATION:
                metrics = ['accuracy']
            elif self.task in consts.TASK_LIST_DETECTION:
                metrics = ['f1']

        y_pred = self.predict(X, verbose=verbose)

        if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_DETECTION:
            y_proba = self.predict_proba(X, verbose=verbose)
            if 'binaryclass' in self.task or self.task in consts.TASK_LIST_DETECTION:
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

    def save(self, model_file, external=False):
        if external:
            open_func = open
            if '.model' in model_file:
                model_file = model_file + '_estimator.pkl'
            else:
                model_file = os.path.join(model_file, 'estimator.pkl')
        else:
            open_func = fs.open

        if self.mode == consts.Mode_STATS:
            with open_func(f'{model_file}', 'wb') as output:
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            subself = copy.copy(self)
            if self.cv_models_ is None:
                subself.model.model.save_model(model_file, external=external)
            else:
                for est in subself.cv_models_:
                    est.model.save_model(model_file + '_' + est.group_id, external=external)
            with open_func(f'{model_file}', 'wb') as output:
                pickle.dump(subself, output, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(model_file, mode, external=False):
        if external:
            open_func = open
            if '.model' in model_file:
                model_file = model_file + '_estimator.pkl'
            else:
                model_file = os.path.join(model_file, 'estimator.pkl')
        else:
            open_func = fs.open

        if mode == consts.Mode_STATS:
            with open_func(f'{model_file}', 'rb') as input:
                estimator = pickle.load(input)
        else:
            from hyperts.framework.dl import BaseDeepEstimator
            with open_func(f'{model_file}', 'rb') as input:
                estimator = pickle.load(input)
            if estimator.cv_models_ is None:
                model = BaseDeepEstimator.load_model(model_file, external=external)
                estimator.model.model.model = model
            else:
                for est in estimator.cv_models_:
                    model = BaseDeepEstimator.load_model(model_file + '_' + est.group_id, external=external)
                    est.model.model = model
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
                 timestamp=None,
                 covariates=None,
                 dispatcher=None,
                 callbacks=None,
                 reward_metric='accuracy',
                 discriminator=None,
                 data_cleaner_params=None,
                 use_layer_weight_cache=False):

        self.mode = mode
        self.timestamp = timestamp
        self.covariates= covariates
        self.data_cleaner_params = data_cleaner_params
        if mode == consts.Mode_NAS and use_layer_weight_cache:
            from hyperts.framework.nas import LayerWeightsCache
            self.weights_cache = LayerWeightsCache()
        else:
            self.weights_cache = None

        HyperModel.__init__(self,
                            searcher,
                            dispatcher=dispatcher,
                            callbacks=callbacks,
                            reward_metric=reward_metric,
                            task=task,
                            discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = HyperTSEstimator(task=self.task,
                                     mode=self.mode,
                                     reward_metric=self.reward_metric,
                                     space_sample=space_sample,
                                     timestamp=self.timestamp,
                                     covariates=self.covariates,
                                     data_cleaner_params=self.data_cleaner_params,
                                     weights_cache=self.weights_cache)
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

    def search(self, X, y, X_eval, y_eval, cv=False, num_folds=3, max_trials=3,
               dataset_id=None, trial_store=None, **fit_kwargs):
        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)

        if trial_store is None:
            pass

        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trial_store))

        self._before_search()

        # dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)
        dispatcher = InProcessDispatcher('/models')

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