import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as sklearn_tts
from hypernets.tabular.toolbox import ToolBox

from hyperts.utils import consts, metrics as metrics_
from hyperts.utils.holidays import get_holidays

class TSToolBox(ToolBox):

    @staticmethod
    def DataFrame(data=None, index = None, columns = None, dtype = None, copy = False):
        """Two-dimensional, size-mutable, potentially heterogeneous tabular data.

        Parameters
        ----------
        data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            Dict can contain Series, arrays, constants, or list-like objects.

            .. versionchanged:: 0.23.0
            If data is a dict, column order follows insertion-order for
            Python 3.6 and later.

            .. versionchanged:: 0.25.0
            If data is a list of dicts, column order follows insertion-order
            for Python 3.6 and later.

        index : Index or array-like
            Index to use for resulting frame. Will default to RangeIndex if
            no indexing information part of input data and no index provided.
        columns : Index or array-like
            Column labels to use for resulting frame. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed. If None, infer.
        copy : bool, default False
            Copy data from inputs. Only affects DataFrame / 2d ndarray input.
        """
        return pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

    @staticmethod
    def join_df(df1: pd.DataFrame, df2: pd.DataFrame, on: None):
        """Join columns of another DataFrame.

        Parameters
        ----------
        on : str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the caller and `other`.
        """
        return df1.join(df2.set_index(on), on=on)

    @staticmethod
    def to_datetime(df: pd.DataFrame, **kwargs):
        """Convert argument to datetime.

        """
        return pd.to_datetime(df, **kwargs)

    @staticmethod
    def datetime_format(df: pd.DataFrame, format='%Y-%m-%d %H:%M:%S'):
        """Convert datetime format.

        """
        return pd.to_datetime(df).dt.strftime(format)

    @staticmethod
    def infer_ts_freq(df: pd.DataFrame, ts_name: str = consts.TIMESTAMP):
        """ Infer the frequency of the time series.
        Parameters
        ----------
        ts_name: 'str', time column name.
        """
        return _infer_ts_freq(df, ts_name)

    @staticmethod
    def multi_period_loop_imputer(df: pd.DataFrame, freq: str, offsets: list = None, max_loops: int = 10):
        """Multiple period loop impute NAN.
        Parameters
        ----------
        freq: str
            'S' - second
            'T' - minute
            'H' - hour
            'D' - day
            'M' - month
            'Y','A', A-DEC' - year
        offsets: list, offset lag.
        max_loops: 'int', maximum number of loop imputed.
        """
        if not isinstance(freq, str):
            return df

        if offsets is None and freq in 'W' or 'W-' in freq or 'WOM-' in freq:
            offsets = [-1, -2, -3, -4, 1, 2, 3, 4]
        elif offsets is None and freq in ['M', 'MS', 'BM', 'CBM', 'CBMS']:
            offsets = [-1, -2, -3, -4, 1, 2, 3, 4]
        elif offsets is None and freq in ['SM', '15D', 'SMS']:
            offsets = [-1, -2, -4, -6, -8, 1, 2, 4, 6, 8]
        elif offsets is None and 'Q' in freq or 'Q-' in freq or 'BQ' in freq or 'BQ-' in freq or 'QS-' in freq or 'BQS-':
            offsets = [-1, -4, -8, -12, 1, 4, 8, 12]
        elif offsets is None and freq in ['A', 'Y'] or 'A-' in freq or 'BA-' in freq or 'AS-' in freq or 'BAS-' in freq:
            offsets = [-1, -2, -3, -4, 1, 2, 3, 4]
        elif offsets is None and 'S' in freq or 'T' in freq or 'min' in freq:
            offsets = [-60*4, -60*3, -60*2, -60*1, -1, 1, 60*1, 60*2, 60*3, 60*4]
        elif offsets is None and 'H' in freq:
            offsets = [-24*4, -24*3, -24*2, -24*1, -1, 1, 24*1, 24*2, 24*3, 24*4,
                      -168*4, -168*3, -168*2, -168*1, 168*1, 168*2, 168*3, 168*4]
        elif offsets is None and 'BH' in freq or '8H' in freq:
            offsets = [-8*4, -8*3, -8*2, -8*1, -1, 1, 8*1, 8*2, 8*3, 8*4,
                      -40*4, -40*3, -40*2, -40*1, 40*1, 40*2, 40*3, 40*4]
        elif offsets is None and 'D' in freq:
            offsets = [-1, -7, -7*2, 7*3, -7*4, 1, 7, 7*2, 7*3, 7*4]
        elif offsets is None and freq in ['C', 'B']:
            offsets = [-1, -5, -5*2, 5*3, -5*4, 1, 5, 5*2, 5*3, 5*4]
        elif offsets is None and 'L' in freq or 'U' in freq or 'N' in freq or 'ms' in freq:
            offsets = [-1, -50, -100, -200, -1000, 1, 50, 100, 200, 1000]
        elif offsets == None:
            offsets = [-1, 1]

        offsets = _expand_list(freq=freq, pre_list=offsets)

        values = df.values.copy()
        loop, missing_rate = 0, 1
        while loop < max_loops and missing_rate > 0:
            values, missing_rate = _impute(values, offsets)
            loop += 1
        values[np.where(np.isnan(values))] = np.nanmean(values)

        fill_df = pd.DataFrame(values, columns=df.columns)
        return fill_df

    @staticmethod
    def forward_period_imputer(df: pd.DataFrame, offset: int):
        """ Forward period imputer.
        Parameters
        ----------
        offsets: 'int', offset lag.
        """
        fill_df = df.fillna(df.rolling(window=offset, min_periods=1).agg(lambda x: x.iloc[0]))
        return fill_df

    @staticmethod
    def simple_numerical_imputer(df: pd.DataFrame, mode='mean'):
        """Fill NaN with mean, mode, 0."""
        if mode == 'mean':
            df = df.fillna(df.mean().fillna(0).to_dict())
        elif mode == 'mode':
            df = df.fillna(df.mode().fillna(0).to_dict())
        else:
            df = df.fillna(0)
        return df

    @staticmethod
    def drop_duplicated_ts_rows(df: pd.DataFrame, ts_name: str = consts.TIMESTAMP, keep_data: str = 'last'):
        """Returns without duplicate time series, the last be keeped by default.
        Example:
            TimeStamp      y
            2021-03-01    3.4
            2021-03-02    5.2
            2021-03-03    9.3
            2021-03-03    9.5
            2021-03-04    6.7
            2021-03-05    2.3
            >>
            TimeStamp      y
            2021-03-01    3.4
            2021-03-02    5.2
            2021-03-03    9.5
            2021-03-04    6.7
            2021-03-05    2.3
        """
        assert isinstance(df, pd.DataFrame)
        drop_df = df.drop_duplicates(subset=[ts_name], keep=keep_data)

        return drop_df

    @staticmethod
    def smooth_missed_ts_rows(df: pd.DataFrame, freq: str = None, ts_name: str = consts.TIMESTAMP):
        """Returns full time series.
        Example:
            TimeStamp      y
            2021-03-01    3.4
            2021-03-02    5.2
            2021-03-04    6.7
            2021-03-05    2.3
            >>
            TimeStamp      y
            2021-03-01    3.4
            2021-03-02    5.2
            2021-03-03    NaN
            2021-03-04    6.7
            2021-03-05    2.3
        """
        assert isinstance(df, pd.DataFrame)
        if freq == None:
            freq = _infer_ts_freq(df, ts_name)
        if df[ts_name].dtypes == object:
            df[ts_name] = pd.to_datetime(df[ts_name])
        df = df.sort_values(by=ts_name)
        start, end = df[ts_name].iloc[0], df[ts_name].iloc[-1]

        full_ts = pd.DataFrame(pd.date_range(start=start, end=end, freq=freq), columns=[ts_name])

        smooth_df = full_ts.join(df.set_index(ts_name), on=ts_name)

        return smooth_df

    @staticmethod
    def clip_to_outliers(df: pd.DataFrame, std_threshold: int = 3):
        """Replace outliers above threshold with that threshold.
        Parameters
        ----------
        std_threshold: 'float', the number of standard deviations away from mean to count as outlier.
        """
        assert isinstance(df, pd.DataFrame)
        df_std = df.std(axis=0, skipna=True)
        df_mean = df.mean(axis=0, skipna=True)
        lower = df_mean - (df_std * std_threshold)
        upper = df_mean + (df_std * std_threshold)
        df_outlier = df.clip(lower=lower, upper=upper, axis=1)

        return df_outlier

    @staticmethod
    def nan_to_outliers(df: pd.DataFrame, std_threshold: int = 3):
        """Replace outliers above threshold with that threshold.
        Parameters
        ----------
        std_threshold: 'float', the number of standard deviations away from mean to count as outlier.
        """
        assert isinstance(df, pd.DataFrame)
        df_outlier = df.copy()
        df_std = df.std(axis=0, skipna=True)
        df_mean = df.mean(axis=0, skipna=True)
        outlier_indices = np.abs(df - df_mean) > df_std * std_threshold
        df_outlier = df_outlier.mask(outlier_indices, other=np.nan)

        return df_outlier

    @staticmethod
    def infer_window_size(max_size: int, freq: str):
        """Infer window of neural net.
        Parameters
        ----------
        max_size: int, maximum time window allowed.
        freq: str or DateOffset.
        """
        if freq in 'W' or 'W-' in freq or 'WOM-' in freq:
            window = list(filter(lambda x: x<=max_size, [1, 2, 3, 4, 5, 6, 7]))
        elif freq in ['SM', 'M', 'MS', 'SMS', 'BM', 'CBM', 'CBMS', '15D']:
            window = list(filter(lambda x: x <= max_size, [1, 3, 6, 12, 24]))
        elif 'Q' in freq or 'Q-' in freq or 'BQ' in freq or 'BQ-' in freq or 'QS-' in freq or 'BQS-':
            window = list(filter(lambda x: x <= max_size, [1, 4, 8, 12, 16]))
        elif freq in ['A', 'Y'] or 'A-' in freq or 'BA-' in freq or 'AS-' in freq or 'BAS-' in freq:
            window = list(filter(lambda x: x<=max_size, [1, 3, 6, 12, 24]))
        elif 'S' in freq or 'T' in freq or 'min' in freq:
            window = list(filter(lambda x: x<=max_size, [1, 5, 10, 30, 60]))
        elif 'H' in freq:
            window = list(filter(lambda x: x<=max_size, [1, 6, 12, 24, 48]))
        elif 'BH' in freq or '8H':
            window = list(filter(lambda x: x<=max_size, [1, 4, 8, 16, 24]))
        elif 'D' in freq:
            window = list(filter(lambda x: x<=max_size, [1, 7, 14, 21]))
        elif freq in ['C', 'B']:
            window = list(filter(lambda x: x<=max_size, [1, 5, 10, 15, 20]))
        elif 'L' in freq or 'U' in freq or 'N' in freq or 'ms' in freq:
            window = list(filter(lambda x: x <= max_size, [1, 10, 50, 100, 500, 1000]))
        else:
            window = list(filter(lambda x: x <= max_size, [1, 3, 5, 7, 12, 24]))

        return _expand_list(freq=freq, pre_list=window)


    @staticmethod
    def generate_ts_covariables(start_date, periods, freq='H'):
        """Generate covariates about time.
        Parameters
        ----------
        start_date: 'str' or datetime-like.
            Left bound for generating dates.
        periods: 'int'.
            Number of periods to generate.
        freq: str or DateOffset, default 'H'.
        """
        dstime = pd.date_range(start_date, periods=periods, freq=freq)
        fds = pd.DataFrame(dstime, columns={'TimeStamp'})
        fds['Hour'] = fds['TimeStamp'].dt.hour
        fds['WeekDay'] = fds['TimeStamp'].dt.weekday
        period_dict = {
            23: 0, 0: 0, 1: 0,
            2: 1, 3: 1, 4: 1,
            5: 2, 6: 2, 7: 2,
            8: 3, 9: 3, 10: 3, 11: 3,
            12: 4, 13: 4,
            14: 5, 15: 5, 16: 5, 17: 5,
            18: 6,
            19: 7, 20: 7, 21: 7, 22: 7,
        }
        fds['TimeSegmnet'] = fds['Hour'].map(period_dict)
        fds['MonthStart'] = fds['TimeStamp'].apply(lambda x: x.is_month_start * 1)
        fds['MonthEnd'] = fds['TimeStamp'].apply(lambda x: x.is_month_end * 1)
        fds['SeasonStart'] = fds['TimeStamp'].apply(lambda x: x.is_quarter_start * 1)
        fds['SeasonEnd'] = fds['TimeStamp'].apply(lambda x: x.is_quarter_end * 1)
        fds['Weekend'] = fds['TimeStamp'].apply(lambda x: 1 if x.dayofweek in [5, 6] else 0)
        public_holiday_list = get_holidays(year=int(start_date[:4]))
        public_holiday_list = public_holiday_list['Date'].to_list()
        fds['Date'] = fds['TimeStamp'].apply(lambda x: x.strftime('%Y%m%d'))
        fds['Holiday'] = fds['Date'].apply(lambda x: 1 if x in public_holiday_list else 0)
        fds.drop(['Date'], axis=1, inplace=True)
        return fds

    @staticmethod
    def df_mean_std(data: pd.DataFrame):
        """Get the mean and standard deviation of the data.

        """
        mean = data.mean()
        std = data.std()
        return mean, std

    @staticmethod
    def infer_forecast_interval(forecast, prior_mu, prior_sigma, n: int = 5, prediction_interval: float = 0.9):
        """A corruption of Bayes theorem.
        It will be sensitive to the transformations of the data.

        """
        from scipy.stats import norm

        p_int = 1 - ((1 - prediction_interval) / 2)
        adj = norm.ppf(p_int)
        upper_forecast, lower_forecast = pd.DataFrame(), pd.DataFrame()
        for index, row in forecast.iterrows():
            data_mu = row
            post_mu = ((prior_mu / prior_sigma ** 2) + ((n * data_mu) / prior_sigma ** 2)
                       ) / ((1 / prior_sigma ** 2) + (n / prior_sigma ** 2))
            lower = pd.DataFrame(post_mu - adj * prior_sigma).transpose()
            lower = lower.where(lower <= data_mu, data_mu, axis=1)
            upper = pd.DataFrame(post_mu + adj * prior_sigma).transpose()
            upper = upper.where(upper >= data_mu, data_mu, axis=1)
            lower_forecast = pd.concat([lower_forecast, lower], axis=0)
            upper_forecast = pd.concat([upper_forecast, upper], axis=0)
        lower_forecast.index = forecast.index
        upper_forecast.index = forecast.index
        return upper_forecast, lower_forecast

    @staticmethod
    def from_3d_array_to_nested_df(data: np.ndarray,
                                   columns: str = None,
                                   cells_as_array: bool = False):
        """Convert Numpy ndarray with shape (nb_samples, series_length, nb_variables)
        into nested pandas DataFrame (with time series as numpy array or pandas Series in cells)

        Parameters
        ----------
        data : np.ndarray
            3-dimensional Numpy array to convert to nested pandas DataFrame format
        columns: list-like, default = None
            Optional list of names to use for naming nested DataFrame's columns
        cells_as_array : bool, default = False
            If True, then nested cells contain Numpy array
            If False, then nested cells contain pandas Series

        Returns
        ----------
        df : pd.DataFrame

        References
        ----------
        sktime_data_processing: https://github.com/Riyabelle25/sktime/blob/main/sktime/utils/data_processing.py
        """

        df = pd.DataFrame()
        nb_samples, series_length, nb_variables = data.shape
        cell = np.array if cells_as_array else pd.Series
        if columns is None:
            columns = [f'Var_{i}' for i in range(nb_variables)]
        else:
            if len(columns) != nb_variables:
                raise ValueError(f'The number of column names supplied [{len(columns)}] \
                                   does not match the number of data variables [{nb_variables}].')
        for i, columns_name in enumerate(columns):
            df[columns_name] = [cell(data[j, :, i]) for j in range(nb_samples)]
        return df

    @staticmethod
    def from_nested_df_to_3d_array(data: pd.DataFrame):
        """Convert nested pandas DataFrame (with time series as numpy array or pandas Series in cells)
        into Numpy ndarray with shape (nb_samples, series_length, nb_variables).

        Parameters
        ----------
        data : pd.DataFrame
            Nested pandas DataFrame

        Returns
        -------
        data_3d : np.arrray
            3-dimensional NumPy array

        References
        ----------from_nested_to_3d_numpy
        sktime_data_processing: https://github.com/Riyabelle25/sktime/blob/main/sktime/utils/data_processing.py
        """

        nested_col_mask = [*data.applymap(lambda cell: isinstance(cell, (np.ndarray, pd.Series))).any().values]
        if nested_col_mask.count(True) == len(nested_col_mask):
            res = np.stack(data.applymap(lambda cell: cell.to_numpy() if isinstance(cell, pd.Series) else cell)
                           .apply(lambda row: np.stack(row), axis=1)
                           .to_numpy())
        else:
            raise ValueError
        return res.transpose(0, 2, 1)

    @staticmethod
    def is_nested_dataframe(data: pd.DataFrame):
        """Determines whether data is a nested Dataframe.

        Returns
        -------
        bool : True or False.
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        is_nested = isinstance(data.iloc[0, 0], (np.ndarray, pd.Series))
        return is_dataframe and is_nested

    @staticmethod
    def random_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
        """Split arrays or matrices into random train and test subsets. This
        is a wrapper of scikit-learn's ``train_test_split`` that has shuffle.
        """
        results = sklearn_tts(*arrays,
                              test_size=test_size,
                              train_size=train_size,
                              random_state=random_state,
                              shuffle=shuffle,
                              stratify=stratify)

        return results

    @staticmethod
    def temporal_train_test_split(*arrays,
                                  test_size=None,
                                  train_size=None,
                                  test_horizon=None):
        """Split arrays or matrices into sequential train and test subsets.This
        is a wrapper of scikit-learn's ``train_test_split`` that does not shuffle.

        Parameters
        ----------
        *arrays : sequence of indexables with same length / shape[0] Allowed inputs
        are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        test_size : float, int or None, optional (default=None)
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.
        train_size : float, int, or None, (default=None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
        test_horizon: int or None, (default=None)
            If int, represents the forecast horizon length.
        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.
        """
        test_size = test_horizon if test_horizon != None else test_size
        if test_horizon != None and test_horizon > arrays[0].shape[0]:
            raise ValueError(f'{test_horizon} is greater than data shape {arrays[0].shape[0]}.')

        results = sklearn_tts(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            shuffle=False,
            stratify=None)

        return [pd.DataFrame(item) if isinstance(item, pd.Series) else item for item in results]

    @staticmethod
    def list_diff(p: list, q: list):
        """Gets the difference set of two lists.
        Parameters
        ----------
        p: list.
        q: list.

        Returns
        A list.
        -------
        Example
            p = [1, 2, 3, 4, 5],  q = [2, 4]
            >> list_diff(p, q)
            >> [1, 3, 5]

            p = [1, 2, 3, 4, 5],  q = []
            >> list_diff(p, q)
            >> [1, 2, 3, 4, 5]
        """
        if q is not None and len(q) > 0:
            # return list(set(p).difference(set(q)))
            return list(filter(lambda x: x not in q, p))
        else:
            return p

    metrics = metrics_.Metrics


def _infer_ts_freq(df: pd.DataFrame, ts_name: str = consts.TIMESTAMP):
    """ Infer the frequency of the time series.
    Parameters
    ----------
    ts_name: 'str', time column name.
    """
    dateindex = pd.DatetimeIndex(pd.to_datetime(df[ts_name]))
    freq = pd.infer_freq(dateindex)
    if freq is not None:
        return freq
    else:
        for i in range(len(df)):
            freq = pd.infer_freq(dateindex[i:i + 3])
            if freq != None:
                return freq
    return None

def _impute(values, offsets):
    """ Index slide imputation.
    Parameters
    ----------
    offsets: list, offset lag.
    """
    indices0, indices1 = np.where(np.isnan(values))
    if len(indices0) > 0 and len(indices1) > 0:
        padding = []
        for offset in offsets:
            offset_indices0 = indices0 + offset
            start_bound_limit = np.where(indices0 + offset < 0)
            end_bound_limit = np.where(indices0 + offset > len(values) - 1)
            offset_indices0[start_bound_limit] = indices0[start_bound_limit]
            offset_indices0[end_bound_limit] = indices0[end_bound_limit]
            padding.append(values[(offset_indices0, indices1)])
        values[(indices0, indices1)] = np.nanmean(padding, axis=0)
        missing_rate = np.sum(np.isnan(values)) / values.size
    else:
        missing_rate = 0.
    return values, missing_rate

def _expand_list(freq, pre_list):
    try:
        import re
        s = int(re.findall('\d+', freq)[0])
        return list(map(lambda x: x // s + 1, pre_list))
    except:
        return pre_list