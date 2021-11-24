import numpy as np
import pandas as pd

import datetime
import chinese_calendar
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split as sklearn_tts

class offsets_pool:
    neighbor  = [-1, 1]
    second    = [-1, 1, -60*4,-60*3,-60*2,-60*1, 60*1,60*2,60*3,60*4]
    minute    = [-1, 1, -60*4,-60*3,-60*2,-60*1, 60*1,60*2,60*3,60*4]
    hour      = [-1, 1, -24*4,-24*3,-24*2,-24*1, 24*1,24*2,24*3,24*4,
                -168*4,-168*3,-168*2,-168*1, 168*1,168*2,168*3,168*4]
    day       = [-1, 1, -7*4, -7*3, -7*2, -7*1, 7*1, 7*2, 7*3, 7*4]
    month     = [-1, 1, -12*4,-12*3,-12*2,-12*1, 12*1,12*2,12*3,12*4]
    year      = [-1, 1]


def reduce_memory_usage(df: pd.DataFrame, verbose=True):
    '''Reduce RAM Usage
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def infer_ts_freq(df: pd.DataFrame, ts_name: str = 'TimeStamp'):
    dateindex = pd.DatetimeIndex(pd.to_datetime(df[ts_name]))
    for i in range(len(df)):
        freq = pd.infer_freq(dateindex[i:i + 3])
        if freq != None:
            return freq


def _inpute(values, offsets):
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


def multi_period_loop_imputer(df: pd.DataFrame, freq: str, offsets: list = None, max_loops: int = 10):
    """Multiple Period Loop Impute NAN.
    Args:
        offsets: list
        freq: str
            'S' - second
            'T' - minute
            'H' - hour
            'D' - day
            'M' - month
            'Y','A', A-DEC' - year
    """
    if offsets == None and freq == 'S':
        offsets = offsets_pool.minute
    elif offsets == None and freq == 'T':
        offsets = offsets_pool.minute
    elif offsets == None and freq == 'H':
        offsets = offsets_pool.hour
    elif offsets == None and freq == 'D':
        offsets = offsets_pool.day
    elif offsets == None and freq == 'M':
        offsets = offsets_pool.month
    elif offsets == None and freq == 'Y':
        offsets = offsets_pool.year
    elif offsets == None:
        offsets = offsets_pool.neighbor

    values = df.values.copy()
    loop, missing_rate = 0, 1
    while loop < max_loops and missing_rate > 0:
        values, missing_rate = _inpute(values, offsets)
        loop += 1
    values[np.where(np.isnan(values))] = np.nanmean(values)

    fill_df = pd.DataFrame(values, columns=df.columns)
    return fill_df


def forward_period_imputer(df: pd.DataFrame, offset: int):
    fill_df = df.fillna(df.rolling(window=offset, min_periods=1).agg(lambda x: x.iloc[0]))
    return fill_df


def simple_numerical_imputer(df: pd.DataFrame, mode='mean'):
    """Fill NaN with mean, mode, 0."""
    if mode == 'mean':
        df = df.fillna(df.mean().fillna(0).to_dict())
    elif mode == 'mode':
        df = df.fillna(df.mode().fillna(0).to_dict())
    else:
        df = df.fillna(0)
    return df


def columns_ordinal_encoder(df: pd.DataFrame):
    enc = OrdinalEncoder(dtype=np.int)
    encoder_df = enc.fit_transform(df)
    return encoder_df


def drop_duplicated_ts_rows(df: pd.DataFrame, ts_name: str = 'TimeStamp', keep_data: str = 'last'):
    """Returns without duplicate time series,  the last be keeped by default.
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


def smooth_missed_ts_rows(df: pd.DataFrame, freq: str = None, ts_name: str = 'TimeStamp'):
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
        freq = infer_ts_freq(df, ts_name)
    if df[ts_name].dtypes == object:
        df[ts_name] = pd.to_datetime(df[ts_name])
    df = df.sort_values(by=ts_name)
    start, end = df[ts_name].iloc[0], df[ts_name].iloc[-1]

    full_ts = pd.DataFrame(pd.date_range(start=start, end=end, freq=freq), columns=[ts_name])

    smooth_df = full_ts.join(df.set_index(ts_name), on=ts_name)

    return smooth_df


def clip_to_outliers(df: pd.DataFrame, std_threshold: int = 3):
    """Replace outliers above threshold with that threshold.
    Args:
        df (pandas.DataFrame): DataFrame containing numeric data
        std_threshold (float): The number of standard deviations away from mean to count as outlier.
    """
    assert isinstance(df, pd.DataFrame)
    df_std = df.std(axis=0, skipna=True)
    df_mean = df.mean(axis=0, skipna=True)
    lower = df_mean - (df_std * std_threshold)
    upper = df_mean + (df_std * std_threshold)
    df_outlier = df.clip(lower=lower, upper=upper, axis=1)

    return df_outlier


def nan_to_outliers(df: pd.DataFrame, std_threshold: int = 3):
    """Replace outliers above threshold with that threshold.
    Args:
        df (pandas.DataFrame): DataFrame containing numeric data
        std_threshold (float): The number of standard deviations away from mean to count as outlier.
    """
    assert isinstance(df, pd.DataFrame)
    df_outlier = df.copy()
    df_std = df.std(axis=0, skipna=True)
    df_mean = df.mean(axis=0, skipna=True)
    outlier_indices = np.abs(df - df_mean) > df_std * std_threshold
    df_outlier = df_outlier.mask(outlier_indices, other=np.nan)

    return df_outlier


def get_holidays(year=None, include_weekends=True):
    """
    :param year: which year
    :param include_weekends: False for excluding Saturdays and Sundays
    :return: list
    """
    if not year:
        year = datetime.datetime.now().year
    else:
        year = year
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    holidays = chinese_calendar.get_holidays(start, end, include_weekends)
    holidays = pd.DataFrame(holidays, columns=['Date'])
    holidays['Date'] = holidays['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    return holidays


def generate_ts_covariables(start_date, periods, freq='H'):
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


def infer_forecast_interval(train, forecast, n: int = 5, prediction_interval: float = 0.9):
    """A corruption of Bayes theorem.
    It will be sensitive to the transformations of the data."""
    prior_mu = train.mean()
    prior_sigma = train.std()
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


def from_3d_array_to_nested_df(data: np.ndarray,
                               columns_names: str = None,
                               cells_as_array: bool = True):
    """Convert Numpy ndarray with shape (nb_samples, series_length, nb_variables)
    into nested pandas DataFrame (with time series as numpy array or pandas Series in cells)
    Parameters
    ----------
    X : np.ndarray
        3-dimensional Numpy array to convert to nested pandas DataFrame format
    column_names: list-like, default = None
        Optional list of names to use for naming nested DataFrame's columns
    cells_as_numpy : bool, default = False
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
    if columns_names is None:
        columns_names = [f'Var_{i}' for i in range(nb_variables)]
    else:
        if len(columns_names) != nb_variables:
            raise ValueError(f'The number of column names supplied [{len(columns_names)}] \
                               does not match the number of data variables [{nb_variables}].')
    for i, columns_name in enumerate(columns_names):
        df[columns_name] = [cell(data[j, :, i]) for j in range(nb_samples)]
    return df


def from_nested_df_to_3d_array(data: pd.DataFrame):
    """Convert nested pandas DataFrame (with time series as numpy array or pandas Series in cells)
    into Numpy ndarray with shape (nb_samples, series_length, nb_variables).
    Parameters
    ----------
    X : pd.DataFrame
        Nested pandas DataFrame
    Returns
    -------
    X_3d : np.arrray
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


def random_train_test_split(*arrays,
                     test_size=None,
                     train_size=None,
                     random_state=None,
                     shuffle=True,
                     stratify=None):
    """Split arrays or matrices into random train and test subsets. This
    is a wrapper of scikit-learn's ``train_test_split`` that has shuffle.
    """
    return sklearn_tts(*arrays,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify)


def temporal_train_test_split(*arrays,
                     test_size=None,
                     train_size=None,
                     test_horizion=None):
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
    test_horizion: int or None, (default=None)
        If int, represents the forecast horizon length.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    test_size = test_horizion if test_horizion != None else test_size
    if test_horizion != None and test_horizion > arrays[0].shape[0]:
        raise ValueError(f'{test_horizion} is greater than data shape {arrays[0].shape[0]}.')
    return sklearn_tts(
        *arrays,
        test_size=test_size,
        train_size=train_size,
        shuffle=False,
        stratify=None)

def list_diff(p: list, q: list):
    """Gets the difference set of two lists.
    Parameters
    p: list.
    q: list.
    ----------
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
    if q is not None and len(q)>0:
        return list(set(p).difference(set(q)))
    else:
        return p
