# -*- coding:utf-8 -*-

import copy
import time
import collections

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from hypernets.utils import logging
from hypernets.tabular import sklearn_ex

from hyperts.utils._base import get_tool_box
from hyperts.utils.transformers import CategoricalTransformer

logger = logging.get_logger(__name__)


class CategoricalColumn(
    collections.namedtuple('CategoricalColumn',
                           ['name',
                            'vocabulary_size',
                            'embedding_dim',
                            'dtype',
                            'input_name'])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, dtype='int32', input_name=None):
        if input_name is None:
            input_name = 'cat_' + name
        if embedding_dim == 0:
            embedding_dim = int(round(vocabulary_size ** 0.25))
        return super(CategoricalColumn, cls).__new__(cls, name, vocabulary_size, embedding_dim, dtype, input_name)


class ContinuousColumn(
    collections.namedtuple('CotinuousColumn',
                           ['name',
                            'column_names',
                            'input_dim',
                            'dtype',
                            'input_name'])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, column_names, input_dim=0, dtype='float32', input_name=None):
        input_dim = len(column_names)
        return super(ContinuousColumn, cls).__new__(cls, name, column_names, input_dim, dtype, input_name)


class MetaPreprocessor:
    """Abstract base class representing Meta Preprocessor.

    """
    def __init__(self):
        self.labels_ = None
        self.classes_ = None
        self.cont_column_names = None
        self.cat_column_names = None

    @property
    def pos_label(self):
        if self.labels_ is not None and len(self.labels_) == 2:
            return self.labels_[1]
        else:
            return None

    @property
    def labels(self):
        return self.labels_

    @property
    def transformers(self):
        return sklearn_ex

    def fit_transform(self, X, y, copy_data=True):
        raise NotImplementedError(
            'fit_transform is a protected abstract method, it must be implemented.'
        )

    def transform_X(self, X, copy_data=True):
        raise NotImplementedError(
            'transform_X is a protected abstract method, it must be implemented.'
        )

    def transform_y(self, y, copy_data=True):
        raise NotImplementedError(
            'transform_y is a protected abstract method, it must be implemented.'
        )

    def transform(self, X, y, copy_data=True):
        raise NotImplementedError(
            'transform is a protected abstract method, it must be implemented.'
        )

    def inverse_transform_y(self, y_indicator):
        raise NotImplementedError(
            'inverse_transform_y is a protected abstract method, it must be implemented.'
        )

    def get_categorical_columns(self):
        return [c.name for c in self.categorical_columns]

    def get_continuous_columns(self):
        cont_vars = []
        for c in self.continuous_columns:
            cont_vars = cont_vars + c.column_names
        return cont_vars

    def _copy(self, obj):
        return copy.deepcopy(obj)

    def _get_shape(self, obj):
        return obj.shape

    def _nunique(self, y):
        return len(y.unique())

    def _append_categorical_cols(self, cols):
        logger.debug(f'{len(cols)} categorical variables appended.')

        if self.categorical_columns is None:
            self.categorical_columns = []

        if cols is not None and len(cols) > 0:
            self.categorical_columns = self.categorical_columns + \
                                       [CategoricalColumn(name,
                                                          voc_size,
                                                          self.embedding_output_dim
                                                          if self.embedding_output_dim > 0
                                                          else min(4 * int(pow(voc_size, 0.25)), 20))
                                        for name, voc_size in cols]

    def _append_continuous_cols(self, cols, input_name):
        if self.continuous_columns is None:
            self.continuous_columns = []
        if cols is not None and len(cols) > 0:
            self.continuous_columns = self.continuous_columns + [ContinuousColumn(name=input_name,
                                                                                  column_names=[c for c in cols])]


class MetaTSFprocessor(MetaPreprocessor):
    """Mata Time Series Forecast Processor.

    Parameters
    ----------
    timestamp: str, time column name (DataFrame).
    embedding_output_dim: int, default 4.
        Embed dimension when there are categorical variables.
    auto_categorize: bool, default False.
    auto_encode_label: bool, default True.
    cat_remain_numeric: bool, default True.
    """
    def __init__(self,
                 timestamp,
                 embedding_output_dim=4,
                 auto_categorize=False,
                 auto_encode_label=True,
                 cat_remain_numeric=True
                 ) -> None:
        super(MetaTSFprocessor, self).__init__()
        self.timestamp = timestamp
        self.embedding_output_dim = embedding_output_dim
        self.auto_categorize = auto_categorize
        self.auto_encode_label = auto_encode_label
        self.cat_remain_numeric = cat_remain_numeric

        self.time_variables = None
        self.target_columns = None
        self.covariable_columns = None
        self.categorical_columns = None
        self.continuous_columns = None
        self.X_transformers = collections.OrderedDict()

    def _validate_fit_transform(self, X, y):
        """Verify that the data conforms to fit_transform.

        """
        if X is None:
            raise ValueError(f'X cannot be none.')
        if y is None:
            raise ValueError(f'y cannot be none.')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        X_shape = self._get_shape(X)
        y_shape = self._get_shape(y)

        if len(X_shape) != 2 or len(y_shape) != 2:
            raise ValueError(f'x and y must be a 2D datasets.')
        if X_shape[0] != y_shape[0]:
            raise ValueError(f"The number of samples of X and y must be the same. X.shape:{X.shape}, y.shape{y.shape}")

    def _concate_Xy(self, X, y):
        """Concat X and y.

        """
        self.covariable_columns = X.columns.tolist()
        self.covariable_columns.remove(self.timestamp)
        self.target_columns = y.columns.tolist()
        self.classes_ = len(y.columns)
        Xy = pd.concat([y, X], axis=1)
        self.time_variables = Xy.pop(self.timestamp)
        return Xy

    def _decouple_Xy(self, Xy):
        """Decouple X and y.

        """
        Xy.insert(0, self.timestamp, self.time_variables)
        X = Xy[[self.timestamp] + self.covariable_columns]
        y = Xy[self.target_columns]
        return X, y

    def _prepare_columns(self, X):
        """Checks for duplicate column names or reindexes object columns.

        """
        if len(set(X.columns)) != len(list(X.columns)):
            cols = [item for item, count in collections.Counter(X.columns).items() if count > 1]
            raise ValueError(f'Columns with duplicate names in X: {cols}')
        if X.columns.dtype != 'object':
            X.columns = ['x_' + str(c) for c in X.columns]
            logger.warn(f"Column index of X has been converted: {X.columns}")
        return X

    def _is_discrete(self, X):
        """Determines whether data is a discrete value.

        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        fractional_part = np.modf(X)[0]
        is_discrete = sum(fractional_part == 0.) == len(X)
        return is_discrete

    def _prepare_features(self, X):
        """Identify the column type and transform.

        """
        start = time.time()

        logger.info(f'Preparing features...')
        num_vars = []
        convert2cat_vars = []
        cat_vars = []

        X_shape = self._get_shape(X)
        unique_upper_limit = round(X_shape[0] ** 0.5)
        for c in X.columns:
            nunique = self._nunique(X[c])
            dtype = str(X[c].dtype)

            if dtype == 'object' or dtype == 'category' or dtype == 'bool':
                cat_vars.append((c, dtype, nunique))
            elif self._is_discrete(X[c]) and nunique < unique_upper_limit:
                cat_vars.append((c, dtype, nunique))
            elif self.auto_categorize and nunique < unique_upper_limit:
                convert2cat_vars.append((c, dtype, nunique))
            else:
                num_vars.append((c, dtype, nunique))

        if len(convert2cat_vars) > 0:
            cat_columns = [c for c, d, n in convert2cat_vars]
            ce = self.transformers.CategorizeEncoder(cat_columns, self.cat_remain_numeric)
            X = ce.fit_transform(X)
            self.X_transformers['categorize'] = ce
            if self.cat_remain_numeric:
                cat_vars = cat_vars + ce.new_columns
                num_vars = num_vars + convert2cat_vars
            else:
                cat_vars = cat_vars + convert2cat_vars
            self.covariable_columns.append(convert2cat_vars[0])

        logger.debug(f'{len(cat_vars)} categorical variables and {len(num_vars)} continuous variables found. '
                     f'{len(convert2cat_vars)} of them are from continuous to categorical.')
        self._append_categorical_cols([(c[0], c[2] + 2) for c in cat_vars])
        self._append_continuous_cols([c[0] for c in num_vars], 'input_continuous_vars_all')
        logger.info(f'Preparing features taken {time.time() - start}s')
        return X

    def _categorical_encoding(self, X):
        """Categorical variables encoding.

        """
        start = time.time()
        logger.info('Categorical encoding...')
        cat_cols = self.get_categorical_columns()
        mle = self.transformers.MultiLabelEncoder(cat_cols)
        X = mle.fit_transform(X)
        self.X_transformers['label_encoder'] = mle
        logger.info(f'Categorical encoding taken {time.time() - start}s')
        return X

    def transform_X(self, X, copy_data=True):
        """Transform X.

        """
        start = time.time()
        logger.info("Transform [X]...")
        if copy_data:
            X = self._copy(X)
        X = self._prepare_columns(X)
        steps = [step for step in self.X_transformers.values()]
        pipeline = make_pipeline(*steps)
        X_t = pipeline.transform(X)
        logger.info(f'transform_X taken {time.time() - start}s')
        return X_t

    def fit_transform(self, X, y, copy_data=True):
        """Fit and Transform.

        """
        start = time.time()

        self._validate_fit_transform(X, y)
        if copy_data:
            X = self._copy(X)
            y = self._copy(y)

        df = self._concate_Xy(X, y)
        df = self._prepare_columns(df)
        df = self._prepare_features(df)

        if self.auto_encode_label:
            df[self.covariable_columns] = self._categorical_encoding(df[self.covariable_columns])
        self.X_transformers['last'] = self.transformers.PassThroughEstimator()

        self.cont_column_names = self.get_continuous_columns()
        self.cat_column_names = self.get_categorical_columns()
        if len(self.cont_column_names) > 0:
            df[self.cont_column_names] = df[self.cont_column_names].astype('float')
        if len(self.cat_column_names) > 0:
            df[self.cat_column_names] = df[self.cat_column_names].astype('category')
        X, y = self._decouple_Xy(df)

        logger.info(f'fit_transform taken {time.time() - start}s')

        return X, y

    def transform(self, X, y, copy_data=True):
        """Transform.

        """
        start = time.time()
        df = self._concate_Xy(X, y)
        df = self._prepare_columns(df)
        if self.covariable_columns is not None:
            df[self.covariable_columns] = self.transform_X(df[self.covariable_columns], copy_data)

        if len(self.cont_column_names) > 0:
            df[self.cont_column_names] = df[self.cont_column_names].astype('float')
        if len(self.cat_column_names) > 0:
            df[self.cat_column_names] = df[self.cat_column_names].astype('category')
        X, y = self._decouple_Xy(df)

        logger.info(f'transform taken {time.time() - start}s')

        return X, y


class MetaTSCprocessor(MetaPreprocessor):
    """Mata Time Series Classification or Regression Processor.

    Parameters
    ----------
    embedding_output_dim: int, default 4.
        Embed dimension when there are categorical variables.
    auto_categorize: bool, default False.
    auto_encode_label: bool, default True.
    cat_remain_numeric: bool, default True.
    """
    def __init__(self,
                 embedding_output_dim=4,
                 auto_categorize=False,
                 auto_discard_unique=True,
                 cat_remain_numeric=True
                 ) -> None:
        super(MetaTSCprocessor, self).__init__()
        self.embedding_output_dim = embedding_output_dim
        self.auto_categorize = auto_categorize
        self.auto_discard_unique = auto_discard_unique
        self.cat_remain_numeric = cat_remain_numeric

        self.y_label_encoder = None
        self.categorical_columns = None
        self.continuous_columns = None

    def _validate_fit_transform(self, X, y):
        """Verify that the data conforms to fit_transform.

        """
        if X is None:
            raise ValueError(f'X cannot be none.')
        if y is None:
            raise ValueError(f'y cannot be none.')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        X_shape = self._get_shape(X)
        y_shape = self._get_shape(y)

        if len(X_shape) != 2:
            raise ValueError(f'X must be a 2D datasets.')
        if X_shape[0] != y_shape[0]:
            raise ValueError(f"The number of samples of X and y must be the same. X.shape:{X.shape}, y.shape{y.shape}")

    def transform_X(self, X, copy_data=False):
        """Transform X.

        """
        tb = get_tool_box(X)
        logger.info("Transform [X]...")
        start = time.time()
        if copy_data:
            X = self._copy(X)
        if tb.is_nested_dataframe(X):
            X = tb.from_nested_df_to_3d_array(X)
        logger.info(f'transform_X taken {time.time() - start}s')
        return X

    def fit_transform_y(self, y):
        """Fit and Transform y.
        Transform ont hot encoding for multiclass.
        """
        if y.dtype != 'float':
            self.y_label_encoder = CategoricalTransformer()
            y = self.y_label_encoder.fit_transform(y)
            self.labels_ = self.y_label_encoder.classes_
            self.classes_ = len(self.labels_)
        else:
            self.labels_ = []
            self.classes_ = 1
        return y

    def transform_y(self, y, copy_data=False):
        """Transform y.
        Transform ont hot encoding for multiclass.
        """
        logger.info("Transform [y]...")
        start = time.time()
        if copy_data:
            y = self._copy(y)
        if self.y_label_encoder is not None:
            y = self.y_label_encoder.transform(y)
        logger.info(f'transform_y taken {time.time() - start}s')
        return y

    def inverse_transform_y(self, y_indicator):
        """Inverse origonal target format.

        """
        if self.y_label_encoder is not None:
            return self.y_label_encoder.inverse_transform(y_indicator)
        else:
            return y_indicator

    def _prepare_features(self, X):
        """Identify the column type and transform.

        """
        start = time.time()

        logger.info(f'Preparing features...')

        num_vars = []
        convert2cat_vars = []
        cat_vars = []

        Series_shape = self._get_shape(X.iloc[0, 0])
        unique_upper_limit = round(Series_shape[0] ** 0.5)
        for c in X.columns:
            nunique = self._nunique(X[c].iloc[0])
            dtype = str(X[c].iloc[0].dtype)

            if nunique <= 1 and self.auto_discard_unique:
                continue

            if dtype == 'object' or dtype == 'category' or dtype == 'bool':
                cat_vars.append((c, dtype, nunique))
            elif self.auto_categorize and nunique < unique_upper_limit:
                convert2cat_vars.append((c, dtype, nunique))
            else:
                num_vars.append((c, dtype, nunique))

        logger.debug(f'{len(cat_vars)} categorical variables and {len(num_vars)} continuous variables found. '
                     f'{len(convert2cat_vars)} of them are from continuous to categorical.')
        self._append_categorical_cols([(c[0], c[2] + 2) for c in cat_vars])
        self._append_continuous_cols([c[0] for c in num_vars], 'input_continuous_all')
        logger.info(f'Preparing features taken {time.time() - start}s')

        return X

    def fit_transform(self, X, y, copy_data=True):
        """Fit and Transform.

        """
        start = time.time()

        self._validate_fit_transform(X, y)
        if copy_data:
            X = self._copy(X)
            y = self._copy(y)

        X = self._prepare_features(X)
        X = self.transform_X(X)
        y = self.fit_transform_y(y)

        self.cont_column_names = self.get_continuous_columns()
        self.cat_column_names = self.get_categorical_columns()

        logger.info(f'fit_transform taken {time.time() - start}s')

        return X, y

    def transform(self, X, y, copy_data=True):
        """Transform.

        """
        start = time.time()
        X = self._prepare_features(X)
        X = self.transform_X(X, copy_data)
        y = self.transform_y(y, copy_data)

        logger.info(f'transform taken {time.time() - start}s')

        return X, y