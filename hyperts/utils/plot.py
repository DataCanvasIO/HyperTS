import numpy as np

from hyperts.utils import get_tool_box, consts
from hypernets.utils import logging
logger = logging.get_logger(__name__)

try:
    import matplotlib.pyplot as plt
    enable_mpl = True
except:
    enable_mpl = False
    logger.error('Importing matplotlib failed. Plotting will not work.')

try:
    import plotly.graph_objects as go
    enable_plotly = True
except:
    enable_plotly = False
    logger.error('Importing plotly failed. Interactive plots will not work.')

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
    plt.show()