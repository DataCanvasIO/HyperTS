import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class PrequentialSplit(_BaseKFold):
    STRATEGY_PREQ_BLS = 'preq-bls'
    STRATEGY_PREQ_SLID_BLS = 'preq-slid-bls'
    STRATEGY_PREQ_BLS_GAP = 'preq-bls-gap'
    """
    Parameters
    ----------
        strategy : Strategies of requential approach applied in blocks for performance estimation
            `preq-bls`:  The data is split into n blocks. In the initial iteration, only the first two blocks 
             are used, the first for training and the second for test. In the next iteration, the second block 
             is merged with the first and the third block is used for test. This procedure continues until all 
             blocks are tested.
            `preq-slid-bls`: Instead of merging the blocks after each iteration (growing window), one can forget 
             the older blocks in a sliding window fashion. This idea is typically adopted when past data becomes 
             deprecated, which is common in non-stationary environments.
            `preq-bls-gap`: This illustrates a prequential approach applied in blocks, where a gap block is 
             introduced. The rationale behind this idea is to increase the independence between training and 
             test sets.
            
        n_splits : int, default=5. 
            Number of splits. Must be at least 2.
            
        max_train_size : int, default=None.
            Maximum size for a single training set. 
        
        test_size : int, default=None.
            Number of samples in each test set. Defaults to
            ``(n_samples - base_size) / (n_splits + 1)``.
            
        gap_size : int, default=0. For strategy `preq-bls`. 
            Number of samples to exclude from the end of each train set before the test set.   
                        
    References
    ----------
        Cerqueira V, Torgo L, Mozetič I. Evaluating time series forecasting models: An empirical study on performance 
        estimation methods[J]. Machine Learning, 2020, 109(11): 1997-2028.
    """

    def __init__(self, strategy='preq-bls', base_size=None, n_splits=5, stride=1, *, max_train_size=None,
                 test_size=None, gap_size=0):
        super(PrequentialSplit, self).__init__(n_splits=max(n_splits, 2), shuffle=False, random_state=None)

        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.base_size = base_size
        self.stride = stride
        self.n_folds = n_splits
        self.strategy = strategy
        self.fold_size = None

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap_size = self.gap_size

        base = 0
        if self.base_size is not None and self.base_size > 0:
            base = self.base_size
        base += n_samples % n_folds

        if self.test_size is not None and self.test_size > 0:
            test_size = self.test_size
        else:
            test_size = (n_samples - base) // n_folds
        self.test_size = test_size

        if self.n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n_samples))

        first_test = n_samples - test_size*n_splits
        if first_test < 0:
            raise ValueError(
                ("Too many splits={0} for number of samples"
                 "={1} with test_size={2}").format(n_splits, n_samples, test_size))

        indices = np.arange(n_samples)
        logger.info(f'n_folds:{self.n_folds}')
        logger.info(f'test_size:{test_size}')
        if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
            test_starts = range(first_test * 2 + base, n_samples, test_size)
        else:
            test_starts = range(first_test + base, n_samples, test_size)
        last_step = -1
        for fold, test_start in enumerate(test_starts):
            if last_step == fold // self.stride:
                # skip this fold
                continue
            else:
                last_step = fold // self.stride
            if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS:
                train_end = test_start - gap_size
                if self.max_train_size and self.max_train_size < train_end:
                    yield (indices[train_end - self.max_train_size:train_end],
                           indices[test_start:test_start + test_size])
                else:
                    yield (indices[:max(train_end, 0)],
                           indices[test_start:test_start + test_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_SLID_BLS:
                if self.max_train_size and self.max_train_size < test_start:
                    yield (indices[test_start - self.max_train_size:test_start],
                           indices[test_start:test_start + test_size])
                else:
                    yield (indices[test_start - (test_size + base):test_start],
                           indices[test_start:test_start + test_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
                yield (indices[:test_start - test_size], indices[test_start:test_start + test_size])
            else:
                raise ValueError(f'{self.strategy} is not supported')