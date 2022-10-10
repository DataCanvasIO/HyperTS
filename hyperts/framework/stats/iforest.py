# -*- coding:utf-8 -*-
"""

"""
import numpy as np
from sklearn.ensemble import IsolationForest
from hyperts.framework.wrappers import BaseAnomalyDetectorWrapper


class TSIsolationForest(BaseAnomalyDetectorWrapper):
    """Isolation Forest for anomaly detection.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float, default=0.05
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.
    """
    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.05,
                 max_features=1.0,
                 bootstrap=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 name='isolation_forest'):
        super(TSIsolationForest, self).__init__(name=name, contamination=contamination)
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _fit(self, X, y=None, **kwargs):
        self.model.fit(X=X, y=None, sample_weight=kwargs.get('sample_weight', None))
        self.decision_scores_ = self.model.decision_function(X) * -1
        self._get_decision_attributes()

    def _predict(self, X, **kwargs):
        decision_func = self.decision_function(X)
        is_outlier = np.zeros_like(decision_func, dtype=int)
        is_outlier[decision_func > self.threshold_] = 1

        return is_outlier

    def decision_function(self, X):
        """Predict anomaly scores for sequences in X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features).

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        self._check_is_fitted()

        if isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        decision_func = self.model.decision_function(X)

        return decision_func * -1
