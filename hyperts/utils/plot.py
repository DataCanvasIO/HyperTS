import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from hyperts.utils import get_tool_box, consts
from hypernets.utils import logging
logger = logging.get_logger(__name__)

try:
    import matplotlib.pyplot as plt
    enable_mpl = True
except:
    enable_mpl = False
    logger.warning('Importing matplotlib failed. Plotting will not work.')

try:
    import plotly.graph_objects as go
    enable_plotly = True
except:
    enable_plotly = False
    logger.warning('Importing plotly failed. Interactive plots will not work.')

def plot_plotly(forecast,
                timestamp_col,
                target_col,
                var_id=0,
                actual=None,
                history=None,
                forecast_interval=None,
                show_forecast_interval=False,
                include_history=False,
                **kwargs):
    """Plots forecast trend curves for the forecst task by plotly.

    Notes
    ----------
    1. This function can plot the curve of only one target variable. If not specified,
    index 0 is ploted by default.

    2. This function supports ploting of historical observations, future actual values,
    and forecast intervals.

    Parameters
    ----------
    forecast: 'DataFrame'. The columns need to include the timestamp column
        and the target columns.
    timestamp_col: str. 'timestamp' column name.
    target_col: str or list. target columns name.
    var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
        it is the name of the target column. default 0.
    actual: 'DataFrame' or None. If it is not None, the column needs to include
        the time column and the target column.
    history: 'DataFrame'. History data. The columns need to include the timestamp column
        and the target columns.
    forecast_interval: 'DataFrame'. Forecast confidence interval.
    show_forecast_interval: 'bool'. Whether to show the forecast intervals.
        Default False.
    include_history: 'bool'. Whether to show the historical observations.
        Default True.

    Returns
    ----------
    fig : 'plotly.graph_objects.Figure'.
    """
    task = kwargs.get('task', consts.Task_FORECAST)
    anomaly_label = kwargs.get('anomaly_label_col')

    tb = get_tool_box(forecast)

    if not isinstance(target_col, list):
        target_col = [target_col]

    if isinstance(var_id, str) and var_id in target_col:
        var_id = target_col.index(var_id)
    elif isinstance(var_id, str) and var_id not in target_col:
        raise ValueError(f'{var_id} might not be target columns {target_col}.')

    if isinstance(timestamp_col, list):
        timestamp_col = timestamp_col[0]

    ts_free = False if timestamp_col in tb.columns_values(forecast) else True

    y_forecast = forecast[target_col] if task in consts.TASK_LIST_FORECAST else forecast[[consts.ANOMALY_LABEL]]
    y_actual = actual[target_col] if actual is not None else None

    if history is not None and include_history:
        y_train = history[target_col]
        if not ts_free:
            X_train = tb.to_datetime(history[timestamp_col])
            X_forecast =  tb.to_datetime(forecast[timestamp_col])
            train_end_date = history[timestamp_col].iloc[-1]
        else:
            X_train = tb.arange(0, len(history))
            X_forecast = tb.arange(len(history), len(history)+len(forecast))
            train_end_date = None
    else:
        X_train, y_train, train_end_date = None, None, None
        if not ts_free:
            X_forecast =  tb.to_datetime(forecast[timestamp_col])
        else:
            X_forecast = tb.arange(0, len(forecast))

    fig = go.Figure()
    plt.set_loglevel('WARNING')

    if task in consts.TASK_LIST_DETECTION:
        outliers = y_forecast.loc[y_forecast[consts.ANOMALY_LABEL] == 1]
        outlier_index = list(outliers.index)
        outliers_trace = go.Scatter(
            x=X_forecast.loc[outlier_index],
            y=y_actual.values[outlier_index, 0],
            mode='markers',
            line=dict(color='rgba(31, 119, 180, 0.7)'),
            name='Anomaly')
        fig.add_trace(outliers_trace)

        if anomaly_label is not None:
            from hyperts.utils.metrics import _infer_pos_label
            pos_label = _infer_pos_label(actual[anomaly_label])
            actual_outliers = actual.loc[actual[anomaly_label] == pos_label]
            actual_outliers_index = list(actual_outliers.index)
            actual_outliers_trace = plt.scatter(
                x=X_forecast.loc[actual_outliers_index],
                y=y_actual.values[actual_outliers_index, 0],
                mode='markers',
                line=dict(color='rgba(250, 43, 20, 0.7)'),
                name='Ground Truth')
            fig.add_trace(actual_outliers_trace)

        severity_score = forecast[[consts.ANOMALY_CONFIDENCE]]*np.nanmean(y_actual.values[:, var_id])
        severity_trace = go.Scatter(
            x=X_forecast,
            y=severity_score.values[:, 0],
            mode='lines',
            line=dict(width=0.5,
                      color='rgba(31, 119, 180, 0.7)'),
            name='Severity Score')
        fig.add_trace(severity_trace)

    if forecast_interval is not None and show_forecast_interval and task in consts.TASK_LIST_FORECAST:
        upper_forecast, lower_forecast = forecast_interval[0], forecast_interval[1]

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=X_forecast,
            y=lower_forecast.values[:, var_id],
            mode='lines',
            line=dict(
                width=0.0,
                color="rgba(0, 90, 181, 0.5)"),
            legendgroup="interval"
        )
        fig.add_trace(lower_bound)

        upper_bound = go.Scatter(
            name='Upper Bound',
            x=X_forecast,
            y=upper_forecast.values[:, var_id],
            line=dict(
                width=0.0,
                color="rgba(0, 90, 181, 0.5)"),
            legendgroup="interval",
            mode='lines',
            fillcolor='rgba(0, 90, 181, 0.2)',
            fill='tonexty'
        )
        fig.add_trace(upper_bound)

    if actual is not None:
        actual_trace = go.Scatter(
            x=X_forecast,
            y=y_actual.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(250, 43, 20, 0.7)'),
            name='Actual'
        )
        fig.add_trace(actual_trace)

    if task in consts.TASK_LIST_FORECAST:
        forecast_trace = go.Scatter(
            x=X_forecast,
            y=y_forecast.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(31, 119, 180, 0.7)'),
            name='Forecast'
        )
        fig.add_trace(forecast_trace)

    if history is not None and include_history:
        history_trace = go.Scatter(
            x=X_train,
            y=y_train.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.7)'),
            name='Historical'
        )
        fig.add_trace(history_trace)

    if train_end_date is not None and include_history:
        new_layout = dict(
            shapes=[dict(
                type="line",
                xref="x",
                yref="paper",
                x0=train_end_date,
                y0=0,
                x1=train_end_date,
                y1=1,
                line=dict(
                    color="rgba(100, 100, 100, 0.7)",
                    width=1.0)
            )],

            annotations=[dict(
                xref="x",
                x=train_end_date,
                yref="paper",
                y=.95,
                text="Observed End Date",
                showarrow=True,
                arrowhead=0,
                ax=-60,
                ay=0
            )]
        )
        fig.update_layout(new_layout)

    if task in consts.TASK_LIST_FORECAST:
        ylabel = y_forecast.columns[var_id]
    else:
        ylabel = y_actual.columns[var_id]

    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title=ylabel),
        title='Actual vs Forecast' if actual is not None else 'Forecast Curve',
        title_x=0.5,
        showlegend=True,
        width=1000,
        legend={'traceorder': 'reversed'},
        hovermode='x',
    )
    fig.update_layout(layout)

    fig.update_xaxes(rangeslider_visible=True)

    fig.show()


def plot_mpl(forecast,
             timestamp_col,
             target_col,
             var_id=0,
             actual=None,
             history=None,
             forecast_interval=None,
             show_forecast_interval=False,
             include_history=False,
             figsize=None,
             grid=True,
             **kwargs):
    """Plots forecast trend curves for the forecst task by matplotlib.

    Notes
    ----------
    1. This function can plot the curve of only one target variable. If not specified,
    index 0 is ploted by default.

    2. This function supports ploting of historical observations, future actual values,
    and forecast intervals.

    Parameters
    ----------
    forecast: 'DataFrame'. The columns need to include the timestamp column
        and the target columns.
    timestamp_col: str. 'timestamp' column name.
    target_col: str or list. target columns name.
    var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
        it is the name of the target column. default 0.
    actual: 'DataFrame' or None. If it is not None, the column needs to include
        the time column and the target column.
    history: 'DataFrame'. History data. The columns need to include the timestamp column
        and the target columns.
    forecast_interval: 'DataFrame'. Forecast confidence interval.
    show_forecast_interval: 'bool'. Whether to show the forecast intervals.
        Default False.
    include_history: 'bool'. Whether to show the historical observations.
        Default True.
    figsize: (float, float), `figure.figsize` Width, height in inches.
        Default (16, 6).
    grid: 'bool'. Whether to display the grid.
        Default True.

    Returns
    ----------
    fig : 'matpltlib..pyplot.figure'.
    """
    task = kwargs.get('task', consts.Task_FORECAST)
    anomaly_label = kwargs.get('anomaly_label_col')

    tb = get_tool_box(forecast)

    if not isinstance(target_col, list):
        target_col = [target_col]

    if isinstance(var_id, str) and var_id in target_col:
        var_id = target_col.index(var_id)
    elif isinstance(var_id, str) and var_id not in target_col:
        raise ValueError(f'{var_id} might not be target columns {target_col}.')

    if isinstance(timestamp_col, list):
        timestamp_col = timestamp_col[0]

    ts_free = False if timestamp_col in tb.columns_values(forecast) else True

    y_forecast = forecast[target_col] if task in consts.TASK_LIST_FORECAST else forecast[[consts.ANOMALY_LABEL]]
    y_actual = actual[target_col] if actual is not None else None

    if history is not None and include_history:
        y_train = history[target_col]
        if not ts_free:
            X_train = tb.to_datetime(history[timestamp_col])
            X_forecast =  tb.to_datetime(forecast[timestamp_col])
        else:
            X_train = tb.arange(0, len(history))
            X_forecast = tb.arange(len(history), len(history)+len(forecast))
    else:
        X_train, y_train = None, None
        if not ts_free:
            X_forecast =  tb.to_datetime(forecast[timestamp_col])
        else:
            X_forecast = tb.arange(0, len(forecast))

    plt.figure(figsize=figsize if figsize is not None else (16, 6))

    if include_history:
        plt.plot(X_train,
                 y_train.values[:, var_id],
                 c='#808080',
                 label='Historical')

        plt.axvline(X_train.iloc[-1],
                   c='#808080',
                   alpha=0.5)

        max_train = np.nanmax(y_train.values[:, var_id])
        if actual is not None:
            max_actual = np.nanmax(y_actual.values[:, var_id])
        else:
            max_actual = max_train
        if task in consts.TASK_LIST_FORECAST:
            max_forcast = np.nanmax(y_forecast.values[:, var_id])
        else:
            max_forcast = max_train
        high_text = max([max_train, max_actual, max_forcast])
        plt.text(X_train.iloc[-1],
                 high_text,
                 s='Observed End Date',
                 fontsize=12,
                 horizontalalignment='right',
                 c='#808080')

    if task in consts.TASK_LIST_FORECAST:
        plt.plot(X_forecast,
                 y_forecast.values[:, var_id],
                 c='#1E90FF',
                 label='Forecast')

    if actual is not None:
        plt.plot(X_forecast,
                 y_actual.values[:, var_id],
                 c='#FF0000',
                 alpha=0.5,
                 label='Actual')

    if forecast_interval is not None and show_forecast_interval and task in consts.TASK_LIST_FORECAST:
        upper_forecast, lower_forecast = forecast_interval[0], forecast_interval[1]

        plt.fill_between(X_forecast,
                         lower_forecast.values[:, var_id],
                         upper_forecast.values[:, var_id],
                         facecolor='#ADD8E6',
                         alpha=0.5,
                         label='Uncertainty Interval')

    if task in consts.TASK_LIST_DETECTION:
        outliers = y_forecast.loc[y_forecast[consts.ANOMALY_LABEL] == 1]
        outlier_index = list(outliers.index)
        plt.scatter(X_forecast.loc[outlier_index],
                    y_actual.values[outlier_index, var_id],
                    c='#1E90FF',
                    label='Anomaly')

        if anomaly_label is not None:
            from hyperts.utils.metrics import _infer_pos_label
            pos_label = _infer_pos_label(actual[anomaly_label])
            actual_outliers = actual.loc[actual[anomaly_label] == pos_label]
            actual_outliers_index = list(actual_outliers.index)
            plt.scatter(X_forecast.loc[actual_outliers_index],
                        y_actual.values[actual_outliers_index, var_id],
                        c='#FF0000',
                        label='Ground Truth')

        severity_score = forecast[[consts.ANOMALY_CONFIDENCE]]*np.nanmean(y_actual.values[:, var_id])
        plt.plot(
            X_forecast,
            severity_score.values[:, 0],
            c='#1E90FF',
            alpha=0.5,
            linewidth=0.2,
            label='Severity Score')

    plt.title('Actual vs Forecast' if actual is not None else 'Forecast Curve', fontsize=16)
    if task in consts.TASK_LIST_FORECAST:
        plt.ylabel(y_forecast.columns[var_id])
    else:
        plt.ylabel(y_actual.columns[var_id])
    plt.xlabel('Date')
    plt.grid(grid, alpha=0.3)
    plt.legend(fontsize=12, loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    plt.subplots_adjust(right=0.85)
    plt.show()


def plot_seasonality(
        df: pd.DataFrame,
        timestamp: str,
        show_ts_col: str,
):
    """Visual the seasonality period of time series.

    Parameters
    ----------
    df: pd.Dataframe, required
        Evaluation data, including timestamp and shown indicator columns.
        Note taht the data type of timestamp column is datetime.
    timestamp: str, required
        Name of datetime column.
    show_ts_col: str, required
        Shown time series column name.
    save_fig: bool, optional, default is False
        If True, save and return image path.

    Returns
    ----------
    fig : 'matpltlib..pyplot.figure'.
    """
    def _concat_ht(x):
        x.fillna(method='ffill', inplace=True)
        x = x.values
        return np.append(x, x[0])

    tb = get_tool_box(df)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be pd.DataFrame.")

    df_cols_all = df.columns.tolist()
    check_cols = [timestamp, show_ts_col]
    for c in check_cols:
        if c not in df_cols_all:
            raise ValueError(f"{c} does not belong to df.")

    timestamp_dtype = df[timestamp].dtype.name
    if timestamp_dtype not in ['datetime64[ns]', 'datetime', 'datetimetz', 'timedelta']:
        raise ValueError(f"Expect {timestamp} column of df is datetime type, but found {timestamp_dtype}")

    freq = tb.infer_ts_freq(df, timestamp)
    if freq == None:
        return f"The data has no regular frequency and cannot visualize seasonality."

    ts = df[[timestamp, show_ts_col]]
    ts.set_index(timestamp, inplace=True)

    ts_resampled = ts.resample(freq).mean()

    daily_seasonality = ts_resampled.groupby(ts_resampled.index.hour).mean()
    weekly_seasonality = ts_resampled.groupby(ts_resampled.index.day_of_week).mean()
    yearly_seasonality = ts_resampled.groupby(ts_resampled.index.month).mean()

    dict_seasonality = {}

    subplot_num = 1
    if yearly_seasonality.shape[0] == 12:
        subplot_num = subplot_num + 1
    if weekly_seasonality.shape[0] == 7:
        subplot_num = subplot_num + 1
    if daily_seasonality.shape[0] == 24:
        subplot_num = subplot_num + 1

    fig = plt.figure(figsize=(10 ,4*subplot_num))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.subplot(subplot_num, 1, 1)
    plt.plot(ts.index, ts.values, c='#808080')
    plt.xlabel('Date')
    plt.ylabel(f"Original Series [{show_ts_col}]", fontsize=15)
    plt.grid(linestyle="--", alpha=0.3)

    i = 2
    if yearly_seasonality.shape[0] == 12:
        ts_y = _concat_ht(yearly_seasonality)
        x = np.linspace(0, len(yearly_seasonality), len(ts_y), endpoint=True)
        X_Y_Spline = make_interp_spline(x, ts_y)
        X_ts = np.linspace(0, len(yearly_seasonality), 366, endpoint=True)
        Y_ts = X_Y_Spline(X_ts)
        xticks = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.', 'Jan.']
        dict_seasonality['seasonality_yearly'] = [xticks[:-1], list(ts_y[:-1])]

        ax1 = plt.subplot(subplot_num, 1, i)
        plt.plot(X_ts, Y_ts, c='r')
        ax1.set_xticks(np.linspace(0, len(yearly_seasonality), 13, endpoint=True))
        ax1.set_xticklabels(xticks)
        ax1.set_xticks(np.linspace(0, len(yearly_seasonality), 13, endpoint=True))
        ax1.set_xticklabels(xticks)
        plt.xlabel('Day of year')
        plt.ylabel('Seasonality: yearly', fontsize=15)
        plt.grid(linestyle="--", alpha=0.3)
        i+=1

    if weekly_seasonality.shape[0] == 7:
        ts_w = _concat_ht(weekly_seasonality)
        x = np.linspace(0, len(weekly_seasonality), len(ts_w), endpoint=True)
        X_Y_Spline = make_interp_spline(x, ts_w)
        X_ts = np.linspace(0, len(weekly_seasonality), 500, endpoint=True)
        Y_ts = X_Y_Spline(X_ts)
        xticks = ['Sun.', 'Mon.', 'Tue.', 'Weds.', 'Thus.', 'Fri.', 'Sat.', 'Sun.']
        dict_seasonality['seasonality_weekly'] = [xticks[:-1], list(ts_w[:-1])]

        ax2 = plt.subplot(subplot_num, 1, i)
        plt.plot(X_ts, Y_ts, c='g')
        ax2.set_xticks(np.linspace(0, len(weekly_seasonality), 8, endpoint=True))
        ax2.set_xticklabels(xticks)
        plt.xlabel('Day of week')
        plt.ylabel('Seasonality: weekly', fontsize=15)
        plt.grid(linestyle="--", alpha=0.3)
        i+=1

    if daily_seasonality.shape[0] == 24:
        ts_d = _concat_ht(daily_seasonality)
        x = np.linspace(0, len(daily_seasonality), len(ts_d), endpoint=True)
        X_Y_Spline = make_interp_spline(x, ts_d)
        X_ts = np.linspace(0, len(daily_seasonality), 500, endpoint=True)
        Y_ts = X_Y_Spline(X_ts)
        xticks = np.linspace(0, 24, 25, endpoint=True, dtype=int)
        dict_seasonality['seasonality_daily'] = [list(xticks)[:-1], list(ts_d[:-1])]

        ax3 = plt.subplot(subplot_num, 1, i)
        plt.plot(X_ts, Y_ts, c='b')
        ax3.set_xticks(np.linspace(0, len(daily_seasonality), 25, endpoint=True))
        ax3.set_xticklabels(xticks)
        plt.xlabel('Hour of day')
        plt.ylabel('Seasonality: daily', fontsize=15)
        plt.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Seasonality Analysis", y=0.95, fontsize=17)

    plt.show()