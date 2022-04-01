# -*- coding:utf-8 -*-

import joblib
import numpy as np
from sklearn.metrics import get_scorer
from hypernets.tabular.ensemble import GreedyEnsemble
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class TSGreedyEnsemble(GreedyEnsemble):
    """
    References
    ----------
        Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first
        international conference on Machine learning. 2004.
    """

    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527,
                 target_dims=1, scoring='neg_log_loss', ensemble_size=0):
        super(TSGreedyEnsemble, self).__init__(task, estimators, need_fit, n_folds, method, random_state=random_state)
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.ensemble_size = ensemble_size
        self.target_dims = target_dims

        # fitted
        self.weights_ = None
        self.scores_ = None
        self.best_stack_ = None
        self.hits_ = None

    def _score(self, y_ture, y_preds, n_jobs=-1):
        fn = joblib.delayed(self.scorer._score_func)
        paral = joblib.Parallel(n_jobs=n_jobs)
        rs = paral(fn(y_ture, p, **self.scorer._kwargs) for p in y_preds)
        rs = [r * self.scorer._sign for r in rs]
        return rs

    def fit(self, X, y, est_predictions=None):
        assert y is not None
        if est_predictions is not None:
            logger.info('validate oof predictions')
            self._validate_predictions(X, y, est_predictions)
        else:
            assert X is not None
            if self.need_fit:
                logger.info(f'get predictions, need_fit={self.need_fit}')
                est_predictions = self._Xy2predicttions(X, y)
            else:
                logger.info(f'get predictions, need_fit={self.need_fit}')
                est_predictions = self._X2predictions(X)

        logger.info('fit_predictions')
        if self.target_dims > 1:
            weights, scores = self._parallel_fit_predictions(est_predictions, y)
            self.weights_ = np.array(weights).transpose((1, 0))
            self.scores_ = np.array(scores).transpose((1, 0))
        else:
            self.fit_predictions(est_predictions, y)

    def predict(self, X):
        est_predictions = self._X2predictions(X)
        pred = self.predictions2predict(est_predictions)
        if self.task != 'regression' and self.classes_ is not None:
            pred = self._indices2predict(pred)
        return pred

    def predictions2predict(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        weights = np.array(self.weights_)
        if len(predictions.shape) == 3 and self.task == 'binary':
            predictions = predictions[:, :, -1]
        if len(predictions.shape) == 3 and self.task == 'multiclass':
            weights = np.expand_dims(weights, axis=1).repeat(predictions.shape[2], 1)

        proba = np.sum(predictions * weights, axis=1)
        pred = self.proba2predict(proba)
        return pred

    def predictions2predict_proba(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        if self.task == 'multiclass' and self.method == 'hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        weights = np.array(self.weights_)
        if len(predictions.shape) == 3 and self.task in ['binary', 'multiclass']:
            weights = np.expand_dims(weights, axis=1).repeat(predictions.shape[2], 1)

        proba = np.sum(predictions * weights, axis=1)

        if self.task == 'regression':
            return proba
        else:
            # guaranteed to sum to 1.0 over classes
            proba = proba * np.expand_dims(1 / (proba.sum(axis=1)), axis=1).repeat(proba.shape[1], 1)

        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba

    def _parallel_fit_predictions(self, est_predictions, y_true, n_jobs=1):
        fn = joblib.delayed(self._fit_predictions)
        paral = joblib.Parallel(n_jobs=n_jobs)
        res = paral(fn(est_predictions[..., i], y_true.values[..., i]) for i in range(self.target_dims))
        weights = [w for w, _ in res]
        scores  = [s for _, s in res]
        return weights, scores

    def _fit_predictions(self, y_preds, y_true):
        self.fit_predictions(y_preds, y_true)
        return self.weights_, self.scores_

    def _validate_predictions(self, X, y, est_predictions):
        if self.target_dims == 1 and (self.task == 'regression' or self.method == 'hard'):
            est_predictions = np.squeeze(est_predictions)
            assert est_predictions.shape == (len(y), len(self.estimators)), \
                f'shape is not equal, may be a wrong task type. task:{self.task},  ' \
                f'est_predictions.shape: {est_predictions.shape}, ' \
                f'(len(y), len(self.estimators)):{(len(y), len(self.estimators))}'
        else:
            assert len(est_predictions.shape) == 3
            assert est_predictions.shape[0] == len(y)
            assert est_predictions.shape[1] == len(self.estimators)

    def _X2predictions(self, X):
        if self.target_dims == 1 and (self.task == 'regression' or self.method == 'hard'):
            est_predictions = np.zeros((len(X), len(self.estimators)), dtype=np.float64)
        elif self.target_dims > 1 and self.task not in ['binary', 'multiclass']:
            est_predictions = np.zeros((len(X), len(self.estimators), self.target_dims), dtype=np.float64)
        else:
            est_predictions = np.zeros((len(X), len(self.estimators), len(self.classes_)), dtype=np.float64)

        for n, estimator in enumerate(self.estimators):
            if estimator is not None:
                pred = self._estimator_predict(estimator, X)
                if self.target_dims == 1 and self.task == 'regression' and len(pred.shape) > 1:
                    assert pred.shape[1] == 1
                    pred = pred.reshape(pred.shape[0])
                est_predictions[:, n] = pred
        return est_predictions