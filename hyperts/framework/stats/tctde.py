# -*- coding: utf-8 -*-
"""TDE classifiers.

Dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["TemporalDictionaryEnsemble", "IndividualTDE", "histogram_intersection"]

import math
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from numba import njit, types
from numba.typed import Dict
from sklearn.utils import check_random_state

from hyperts.framework.stats.sktime_ex import BaseClassifier
from hyperts.framework.stats.sktime_ex._sfa import SFA


class IndividualTDEClassifier(BaseClassifier):
    """Single TDE classifier, an extension of the Bag of SFA Symbols (BOSS) model.

    Base classifier for the TDE classifier. Implementation of single TDE base model
    from Middlehurst (2021). [1]_

    Overview: input "n" series of length "m" and IndividualTDE performs a SFA
    transform to form a sparse dictionary of discretised words. The resulting
    dictionary is used with the histogram intersection distance function in a
    1-nearest neighbor.

    fit involves finding "n" histograms.

    predict uses 1 nearest neighbor with the histogram intersection distance function.

    Parameters
    ----------
    window_size : int, default=10
        Size of the window to use in the SFA transform.
    word_length : int, default=8
        Length of word to use to use in the SFA transform.
    norm : bool, default=False
        Whether to normalize SFA words by dropping the first Fourier coefficient.
    levels : int, default=1
        The number of spatial pyramid levels for the SFA transform.
    igb : bool, default=False
        Whether to use Information Gain Binning (IGB) or
        Multiple Coefficient Binning (MCB) for the SFA transform.
    alphabet_size : default=4
        Number of possible letters (values) for each word.
    bigrams : bool, default=False
        Whether to record word bigrams in the SFA transform.
    dim_threshold : float, default=0.85
        Accuracy threshold as a propotion of the highest accuracy dimension for words
        extracted from each dimensions. Only applicable for multivariate data.
    max_dims : int, default=20
        Maximum number of dimensions words are extracted from. Only applicable for
        multivariate data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.

    See Also
    --------
    TemporalDictinaryEnsemble, SFA

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/IndividualTDE.java>`_.

    References
    ----------
    ..  [1] Matthew Middlehurst, James Large, Gavin Cawley and Anthony Bagnall
        "The Temporal Dictionary Ensemble (TDE) Classifier for Time Series
        Classification", in proceedings of the European Conference on Machine Learning
        and Principles and Practice of Knowledge Discovery in Databases, 2020.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        window_size=10,
        word_length=8,
        norm=False,
        levels=1,
        igb=False,
        alphabet_size=4,
        bigrams=True,
        dim_threshold=0.85,
        max_dims=20,
        n_jobs=1,
        random_state=None,
    ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.levels = levels
        self.igb = igb
        self.alphabet_size = alphabet_size
        self.bigrams = bigrams

        # multivariate
        self.dim_threshold = dim_threshold
        self.max_dims = max_dims

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0

        self._transformers = []
        self._transformed_data = []
        self._class_vals = []
        self._dims = []
        self._highest_dim_bit = 0
        self._accuracy = 0
        self._subsample = []
        self._train_predictions = []

        super(IndividualTDEClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit a single base TDE classifier on n_instances cases (X,y).

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self._class_vals = y

        # select dimensions using accuracy estimate if multivariate
        if self.n_dims_ > 1:
            self._dims, self._transformers = self._select_dims(X, y)

            words = [defaultdict(int) for _ in range(self.n_instances_)]

            for i, dim in enumerate(self._dims):
                X_dim = X[:, dim, :].reshape(self.n_instances_, 1, self.series_length_)
                dim_words = self._transformers[i].transform(X_dim, y)
                dim_words = dim_words[0]

                for n in range(self.n_instances_):
                    for word, count in dim_words[n].items():
                        words[n][word << self._highest_dim_bit | dim] = count

            self._transformed_data = words
        else:
            self._transformers.append(
                SFA(
                    word_length=self.word_length,
                    alphabet_size=self.alphabet_size,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                    remove_repeat_words=True,
                    lower_bounding=False,
                    save_words=False,
                    use_fallback_dft=True,
                    n_jobs=self._threads_to_use,
                )
            )
            sfa = self._transformers[0].fit_transform(X, y)
            self._transformed_data = sfa[0]

    def _predict(self, X):
        """Predict class values of all instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        num_cases = X.shape[0]

        if self.n_dims_ > 1:
            words = [defaultdict(int) for _ in range(num_cases)]

            for i, dim in enumerate(self._dims):
                X_dim = X[:, dim, :].reshape(num_cases, 1, self.series_length_)
                dim_words = self._transformers[i].transform(X_dim)
                dim_words = dim_words[0]

                for n in range(num_cases):
                    for word, count in dim_words[n].items():
                        words[n][word << self._highest_dim_bit | dim] = count

            test_bags = words
        else:
            test_bags = self._transformers[0].transform(X)
            test_bags = test_bags[0]

        classes = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._test_nn)(
                test_bag,
            )
            for test_bag in test_bags
        )

        return np.array(classes)

    def _test_nn(self, test_bag):
        rng = check_random_state(self.random_state)

        best_sim = -1
        nn = None

        for n, bag in enumerate(self._transformed_data):
            sim = histogram_intersection(test_bag, bag)

            if sim > best_sim or (sim == best_sim and rng.random() < 0.5):
                best_sim = sim
                nn = self._class_vals[n]

        return nn

    def _select_dims(self, X, y):
        self._highest_dim_bit = (math.ceil(math.log2(self.n_dims_))) + 1
        accs = []
        transformers = []

        # select dimensions based on reduced bag size accuracy
        for i in range(self.n_dims_):
            self._dims.append(i)
            transformers.append(
                SFA(
                    word_length=self.word_length,
                    alphabet_size=self.alphabet_size,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                    remove_repeat_words=True,
                    lower_bounding=False,
                    save_words=False,
                    keep_binning_dft=True,
                    use_fallback_dft=True,
                    n_jobs=self._threads_to_use,
                )
            )

            X_dim = X[:, i, :].reshape(self.n_instances_, 1, self.series_length_)

            transformers[i].fit(X_dim, y)
            sfa = transformers[i].transform(
                X_dim,
                y,
            )
            transformers[i].keep_binning_dft = False
            transformers[i].binning_dft = None

            correct = 0
            for i in range(self.n_instances_):
                if self._train_predict(i, sfa[0]) == y[i]:
                    correct = correct + 1

            accs.append(correct)

        max_acc = max(accs)

        dims = []
        fin_transformers = []
        for i in range(self.n_dims_):
            if accs[i] >= max_acc * self.dim_threshold:
                dims.append(i)
                fin_transformers.append(transformers[i])

        if len(dims) > self.max_dims:
            idx = self.random_state.choice(
                len(dims),
                self.max_dims,
                replace=False,
            ).tolist()
            dims = [dims[i] for i in idx]
            fin_transformers = [fin_transformers[i] for i in idx]

        return dims, fin_transformers

    def _train_predict(self, train_num, bags=None):
        if bags is None:
            bags = self._transformed_data

        test_bag = bags[train_num]
        best_sim = -1
        nn = None

        for n, bag in enumerate(bags):
            if n == train_num:
                continue

            sim = histogram_intersection(test_bag, bag)

            if sim > best_sim:
                best_sim = sim
                nn = self._class_vals[n]

        return nn


def histogram_intersection(first, second):
    """Find the distance between two histograms using the histogram intersection.

    This distance function is designed for sparse matrix, represented as a
    dictionary or numba Dict, but can accept arrays.

    Parameters
    ----------
    first : dict, numba.Dict or array
        First dictionary used in distance measurement.
    second : dict, numba.Dict or array
        Second dictionary that will be used to measure distance from `first`.

    Returns
    -------
    dist : float
        The histogram intersection distance between the first and second dictionaries.
    """
    if isinstance(first, dict):
        sim = 0
        for word, val_a in first.items():
            val_b = second.get(word, 0)
            sim += min(val_a, val_b)
        return sim
    elif isinstance(first, Dict):
        return _histogram_intersection_dict(first, second)
    else:
        return np.sum(
            [
                0 if first[n] == 0 else np.min(first[n], second[n])
                for n in range(len(first))
            ]
        )


@njit(fastmath=True)
def _histogram_intersection_dict(first, second):
    sim = 0
    for word, val_a in first.items():
        val_b = second.get(word, types.uint32(0))
        sim += min(val_a, val_b)
    return sim