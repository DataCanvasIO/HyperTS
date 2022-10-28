# -*- coding:utf-8 -*-
"""

"""
import numpy as np
from sklearn.svm import OneClassSVM
from hyperts.framework.wrappers import BaseAnomalyDetectorWrapper


class TSOneClassSVM(BaseAnomalyDetectorWrapper):
    """One-Class Support Vector Mechine for anomaly detection.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=2
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    contamination : 'auto' or float, default=0.05
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    """
    def __init__(self,
                 kernel="rbf",
                 degree=2,
                 gamma="auto",
                 coef0=0.0,
                 tol=1e-3,
                 nu=0.5,
                 shrinking=True,
                 cache_size=200,
                 max_iter=-1,
                 contamination=0.05,
                 verbose=False,
                 name='one class svm'):
        super(TSOneClassSVM, self).__init__(name=name, contamination=contamination)
        self.model = OneClassSVM(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
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
