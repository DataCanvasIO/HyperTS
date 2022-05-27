# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.decomposition import PCA


def scale(x):
    """
    Z-Score.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    scaled = (x - np.nanmean(x)) / np.nanstd(x, ddof=1)

    return scaled

def fft_infer_period(x):
    """Fourier inference period.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.

    References
    ----------
    1. https://github.com/xuawai/AutoPeriod/blob/master/auto_period.ipynb
    """
    try:
        if isinstance(x, pd.DataFrame):
            x = x.values.reshape(-1, )
        ft = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1)
        mags = abs(ft)
        inflection = np.diff(np.sign(np.diff(mags)))
        peaks = (inflection < 0).nonzero()[0] + 1
        peak = peaks[mags[peaks].argmax()]
        signal_freq = freqs[peak]
        period = int(1 / signal_freq)
    except:
        period = 2

    return {'period': period}


def freq_to_numerical(x, timestamp, freq_mapping_dict=None):
    """Calculates frequency.

    Parameters
    ----------
    x: pd.DataFrame, the timestamp.
    timestamp: str, timestamp name of x.
    freq_mapping_dict, dict, default {'H': 24, 'D': 7, 'W': 54, 'M': 12,
        'Q': 4, 'Y': 1, 'A': 1, 'S': 60, 'T': 60}.
    """
    x[timestamp] = pd.to_datetime(x[timestamp])
    x = x.sort_values([timestamp])
    dateindex = pd.DatetimeIndex(x[timestamp])
    sfreq = pd.infer_freq(dateindex)
    if sfreq is None:
        for i in range(len(x)):
            sfreq = pd.infer_freq(dateindex[i:i + 3])
            if sfreq != None:
                break

    if freq_mapping_dict is None:
        freq_mapping_dict =  {
            'H': 24,
            'D': 7,
            'W': 54,
            'M': 12,
            'Q': 4,
            'Y': 1,
            'A': 1,
            'T': 60,
            'S': 60}
    nfreq = freq_mapping_dict.get(sfreq[0], np.nan)

    return {'nfreq': nfreq}, sfreq


def statistics(x, period: int = 1):
    """
    Calculates statistics features, including length, count,
    mean, var, min, max, median, range, hmean, iqr.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_len = x.shape[0]
    x_col = x.shape[1]
    x_mean = np.nanmean(x, axis=0)
    x_var = np.nanvar(x, ddof=1, axis=0)
    x_min = np.nanmin(x, axis=0)
    x_max = np.nanmax(x, axis=0)
    x_median = np.nanmedian(x, axis=0)
    x_range = np.ptp(x, axis=0)
    x_iqr = stats.iqr(x, axis=0)

    stat_features =  {
        'length_series': x_len,
        'count_series': x_col,
        'mean.mean_series': np.nanmean(x_mean),
        'std.mean_series': np.nanstd(x_mean, ddof=1),
        'mean.var_series': np.nanmean(x_var),
        'std.var_series': np.nanstd(x_var, ddof=1),
        'mean.min_series': np.nanmean(x_min),
        'std.min_series': np.nanstd(x_min, ddof=1),
        'mean.max_series': np.nanmean(x_max),
        'std.max_series': np.nanstd(x_max, ddof=1),
        'mean.median_series': np.nanmean(x_median),
        'std.median_series': np.nanstd(x_median, ddof=1),
        'mean.range_series': np.nanmean(x_range),
        'std.range_series': np.nanstd(x_range, ddof=1),
        'mean.iqr_series': np.nanmean(x_iqr),
        'std.iqr_series': np.nanstd(x_iqr, ddof=1)
    }

    return stat_features


def acf_features(x, period: int = 1):
    """
    Calculates autocorrelation function features.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_len = len(x)

    y_acf_list = acf(x, nlags=max(period, 10), fft=False)

    if x_len > 10:
        y_acf_diff1_list = acf(np.diff(x, n=1), nlags=10, fft=False)
    else:
        y_acf_diff1_list = [np.nan] * 2

    if x_len > 11:
        y_acf_diff2_list = acf(np.diff(x, n=2), nlags=10, fft=False)
    else:
        y_acf_diff2_list = [np.nan] * 2

    y_acf1 = y_acf_list[1]
    y_acf10 = np.nansum((y_acf_list[1:11]) ** 2) if x_len > 10 else np.nan

    diff1y_acf1 = y_acf_diff1_list[1]
    diff1y_acf10 = np.nansum((y_acf_diff1_list[1:11]) ** 2) if x_len > 10 else np.nan

    diff2y_acf1 = y_acf_diff2_list[1]
    diff2y_acf10 = np.nansum((y_acf_diff2_list[1:11]) ** 2) if x_len > 11 else np.nan

    seas_acf1 = y_acf_list[period] if len(y_acf_list) > period else np.nan

    acf_features = {
        'y_acf1': y_acf1,
        'y_acf10': y_acf10,
        'diff1y_acf1': diff1y_acf1,
        'diff1y_acf10': diff1y_acf10,
        'diff2y_acf1': diff2y_acf1,
        'diff2y_acf10': diff2y_acf10,
        'seas_acf1': seas_acf1
    }

    return acf_features


def pacf_features(x, period: int = 1):
    """
    Calculates partial autocorrelation function features.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_len = len(x)

    try:
        y_pacf_list = pacf(x, nlags=max(period, 5), method='ldb')
    except:
        y_pacf_list = np.nan

    if x_len > 5:
        try:
            y_pacf5 = np.nansum(y_pacf_list[1:6] ** 2)
        except:
            y_pacf5 = np.nan
    else:
        y_pacf5 = np.nan

    if x_len > 6:
        try:
            diff1y_pacf_list = pacf(np.diff(x, n=1), nlags=5, method='ldb')
            diff1y_pacf5 = np.nansum(diff1y_pacf_list[1:6] ** 2)
        except:
            diff1y_pacf5 = np.nan
    else:
        diff1y_pacf5 = np.nan

    if x_len > 7:
        try:
            diff2_pacf_list = pacf(np.diff(x, n=2), nlags=5, method='ldb')
            diff2y_pacf5 = np.nansum(diff2_pacf_list[1:6] ** 2)
        except:
            diff2y_pacf5 = np.nan
    else:
        diff2y_pacf5 = np.nan

    try:
        seas_pacf1 = y_pacf_list[period]
    except:
        seas_pacf1 = np.nan

    pacf_features = {
        'y_pacf5': y_pacf5,
        'diff1y_pacf5': diff1y_pacf5,
        'diff2y_pacf5': diff2y_pacf5,
        'seas_pacf1': seas_pacf1
    }

    return pacf_features


def crossing_points(x, period: int = 1):
    """
    Crossing points.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    midline = np.median(x)
    ab = x <= midline
    p1 = ab[:(len(x) - 1)]
    p2 = ab[1:]
    cps = (p1 & (~p2)) | (p2 & (~p1))

    return {'crossing_points': cps.sum()}


def stability(x, period: int = 10):
    """
    Calculate the stability of time series.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    width = period if period > 1 else 10

    try:
        meanx = [np.nanmean(x_w) for x_w in np.array_split(x, len(x) // width + 1)]
        stability = np.nanvar(meanx, ddof=1)
    except:
        stability = np.nan

    return {'stability': stability}


def lumpiness(x, period: int = 10):
    """
    Calculating the lumpiness of time series.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    width = period if period > 1 else 10

    try:
        varx = [np.nanvar(x_w, ddof=1) for x_w in np.array_split(x, len(x) // width + 1)]
        lumpiness = np.nanmean(varx)
    except:
        lumpiness = np.nan

    return {'lumpiness': lumpiness}


def entropy(x, period: int = 1):
    """
    Calculates spectral entropy.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    _, psd = periodogram(x, period)

    psd_norm = psd / np.nansum(psd)
    entropy = np.nansum(psd_norm * np.log2(psd_norm))
    spectral_entropy = -(entropy / np.log2(len(psd_norm)))

    return {'spectral_entropy': spectral_entropy}


def hurst(x, period: int = 30):
    """
    Calculates hurst exponet.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_len = len(x)

    try:
        lags = range(2, min(period, x_len - 1))
        tau = [np.nanstd(x[lag:] - x[:-lag]) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] if not np.isnan(poly[0]) else np.nan
    except:
        hurst = np.nan

    return {'hurst_exponet': hurst}


def stl_features(x, period: int = 1):
    """
    Calculates the strength of trend and seasonality, spikiness, linearity,
    curvature, peak, trough, e_acf1.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    3. https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures (Python code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x_len = len(x)

    try:
        stl = STL(x, period=period).fit()

        trend_ = stl.trend
        seasonal_ = stl.seasonal
        remainder_ = stl.resid

        re_var = np.nanvar(remainder_, ddof=1)
        re_mean = np.nanmean(remainder_)

        trend_strength = 1 - re_var / np.nanvar(trend_ + remainder_, ddof=1)

        seasonal_strength = 1 - re_var / np.nanvar(seasonal_ + remainder_, ddof=1)

        d = (remainder_ - re_mean) ** 2
        varloo = (re_var * (x_len - 1) - d) / (x_len - 2)
        spikiness = np.nanvar(varloo, ddof=1)

        time = np.arange(x_len) + 1
        poly_m = np.transpose(np.vstack(list((time ** k for k in range(3)))))
        poly_m = np.linalg.qr(poly_m)[0][:, 1:]
        time_x = add_constant(poly_m)
        coefs = OLS(trend_, time_x).fit().params
        linearity = coefs[1]
        curvature = -coefs[2]

        peak = (np.argmax(seasonal_) + 1) % period
        peak = period if peak == 0 else peak

        trough = (np.argmin(seasonal_) + 1) % period
        trough = period if trough == 0 else trough

        acfremainder = acf_features(remainder_, period)
        e_acf1 = acfremainder['y_acf1']
    except:
        return {
            'trend_strength': np.nan,
            'seasonal_strength': np.nan,
            'spikiness': np.nan,
            'linearity': np.nan,
            'curvature': np.nan,
            'peak': np.nan,
            'trough': np.nan,
            'e_acf1': np.nan
        }
    return {
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength,
        'spikiness': spikiness,
        'linearity': linearity,
        'curvature': curvature,
        'peak': peak,
        'trough': trough,
        'e_acf1': e_acf1
    }


def holt_parameters(x, period: int = 1):
    """
    Calculates the parameters of a Holt model.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    try:
        holt = ExponentialSmoothing(x, trend='add').fit()

        alpha = holt.params['smoothing_level']
        beta = holt.params['smoothing_trend']
    except:
        return {'alpha': np.nan,
                'beta': np.nan
            }

    return {'alpha': alpha,
            'beta': beta
        }


def hw_parameters(x: np.ndarray, period: int = 1):
    """
    Calculates the parameters of a Holt-Winters model.

    Parameters
    ----------
    x: np.array or pd.DataFrame, the time series.
    period: int, the seasonal of the time series.

    References
    ----------
    1. T.S. Talagala, et al., Meta-learning how to forecast time series, 2018.
    2. https://github.com/robjhyndman/tsfeatures (R code)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    try:
        hw = ExponentialSmoothing(
            endog=x,
            seasonal_periods=period,
            seasonal='add',
            trend='add').fit()

        hw_alpha = hw.params['smoothing_level']
        hw_beta = hw.params['smoothing_trend']
        hw_gamma = hw.params['smoothing_seasonal']
    except:
        return {'hw_alpha': np.nan,
                'hw_beta': np.nan,
                'hw_gamma': np.nan
            }

    return {'hw_alpha': hw_alpha,
            'hw_beta': hw_beta,
            'hw_gamma': hw_gamma
        }


ts_metafeatures_list = {
    'acf_features': acf_features,
    'pacf_features': pacf_features,
    'crossing_points': crossing_points,
    'stability': stability,
    'lumpiness': lumpiness,
    'entropy': entropy,
    'hurst': hurst,
    'stl_features': stl_features,
    'holt_parameters': holt_parameters,
    'hw_parameters': hw_parameters
}


def metafeatures_from_timeseries(
        x,
        timestamp,
        period=None,
        scale_ts=True,
        freq_mapping_dict=None,
        features_list=None):
    """
    Extracting the meta-features of time series.

    Parameters
    ----------
    x: pd.DataFrame, the time series.
    timestamp: str, timestamp name of x.
    period: int or None, the seasonal of the time series, default None.
    scale_ts: bool, whether scale original time series.
    freq_mapping_dict, dict, default {'H': 24, 'D': 7, 'W': 54, 'M': 12,
        'Q': 4, 'Y': 1, 'A': 1, 'S': 60, 'T': 60}.
    features_list, List[str], default ['simple', 'all'].
    """
    metafeatures_dict = {}

    if not isinstance(x, pd.DataFrame):
        raise ValueError('x should be a DataFrame')

    if features_list is None:
        features_list = ['simple', 'all']

    nfreq, sfreq = freq_to_numerical(x, timestamp, freq_mapping_dict)
    metafeatures_dict.update(nfreq)

    x = x.drop(columns=[timestamp])

    if period is None:
        val_cols = x.columns.to_list()
        periods = [fft_infer_period(x[col])['period'] for col in val_cols]
        period = int(np.argmax(np.bincount(periods)))
    metafeatures_dict['period'] = period

    if period > 365:
        if isinstance(nfreq['nfreq'], int):
            period = nfreq['nfreq']
        elif isinstance(sfreq[0], int):
            period = sfreq[0]
        else:
            period = 365

    x = np.array(x)

    if scale_ts:
        x = scale(x)

    if 'simple' in features_list:
        features_list.remove('simple')
        stat_mfs = statistics(x)
        metafeatures_dict.update(stat_mfs)

    if 'all' in features_list:
        metafeatures_list = ts_metafeatures_list.keys()
    else:
        metafeatures_list = features_list

    if len(x.shape) == 2 and x.shape[1] != 1:
        x = PCA(n_components=1).fit_transform(x)
    if len(x.shape) != 1:
        x = x.reshape(-1, )

    for mf in metafeatures_list:
        if ts_metafeatures_list.get(mf) is not None:
            feature = ts_metafeatures_list[mf](x, period)
            metafeatures_dict.update(feature)

    metafeatures = pd.DataFrame(metafeatures_dict, index=[0])
    metafeatures.fillna(0, inplace=True)

    return metafeatures