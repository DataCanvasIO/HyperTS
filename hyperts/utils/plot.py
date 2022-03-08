import pandas as pd

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
                include_history=False):
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
    timestamp_col:
    target_col:
    var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
        it is the name of the target column. default 0.
    actual: 'DataFrame' or None. If it is not None, the column needs to include
        the time column and the target column.
    history:
    forecast_interval:

    show_forecast_interval: 'bool'. Whether to show the forecast intervals.
        Default False.
    include_history: 'bool'. Whether to show the historical observations.
        Default True.

    Returns
    ----------
    fig : 'plotly.graph_objects.Figure'.
    """

    if isinstance(var_id, str) and var_id in target_col:
        var_id = target_col.index(var_id)
    elif isinstance(var_id, str) and var_id not in target_col:
        raise ValueError(f'{var_id} might not be target columns {target_col}.')

    if isinstance(timestamp_col, list):
        timestamp_col = timestamp_col[0]

    X_forecast, y_forecast = forecast[[timestamp_col]], forecast[target_col]

    if actual is not None:
        X_test, y_test = actual[[timestamp_col]], actual[target_col]
    else:
        X_test, y_test = None, None

    if history is not None and include_history:
        X_train, y_train = history[[timestamp_col]], history[target_col]
        train_end_date = X_train[timestamp_col].iloc[-1]
    else:
        X_train, y_train, train_end_date = None, None, None

    fig = go.Figure()
    plt.set_loglevel('WARNING')

    if forecast_interval is not None and show_forecast_interval:
        upper_forecast, lower_forecast = forecast_interval[0], forecast_interval[1]

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=pd.to_datetime(X_forecast[timestamp_col]),
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
            x=pd.to_datetime(X_forecast[timestamp_col]),
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
            x=pd.to_datetime(X_test[timestamp_col]),
            y=y_test.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(250, 43, 20, 0.7)'),
            name='Actual'
        )
        fig.add_trace(actual_trace)

    forecast_trace = go.Scatter(
        x=pd.to_datetime(X_forecast[timestamp_col]),
        y=y_forecast.values[:, var_id],
        mode='lines',
        line=dict(color='rgba(31, 119, 180, 0.7)'),
        name='Forecast'
    )
    fig.add_trace(forecast_trace)

    if train_end_date is not None:
        history_trace = go.Scatter(
            x=pd.to_datetime(X_train[timestamp_col]),
            y=y_train.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.7)'),
            name='Historical'
        )
        fig.add_trace(history_trace)

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

    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title=y_forecast.columns[var_id]),
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
             grid=True):
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
    timestamp_col:
    target_col:
    var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
        it is the name of the target column. default 0.
    actual: 'DataFrame' or None. If it is not None, the column needs to include
        the time column and the target column.
    history:
    forecast_interval:

    show_forecast_interval: 'bool'. Whether to show the forecast intervals.
        Default False.
    include_history: 'bool'. Whether to show the historical observations.
        Default True.

    Returns
    ----------
    fig : 'matpltlib..pyplot.figure'.
    """

    if isinstance(var_id, str) and var_id in target_col:
        var_id = target_col.index(var_id)
    elif isinstance(var_id, str) and var_id not in target_col:
        raise ValueError(f'{var_id} might not be target columns {target_col}.')

    if isinstance(timestamp_col, list):
        timestamp_col = timestamp_col[0]

    X_forecast, y_forecast = forecast[[timestamp_col]], forecast[target_col]

    if actual is not None:
        X_test, y_test = actual[[timestamp_col]], actual[target_col]
    else:
        X_test, y_test = None, None

    if history is not None and include_history:
        X_train, y_train = history[[timestamp_col]], history[target_col]
    else:
        X_train, y_train = None, None

    plt.figure(figsize=figsize if figsize is not None else (16, 6))

    if include_history:
        plt.plot(pd.to_datetime(X_train[timestamp_col]),
                 y_train.values[:, var_id],
                 c='#808080',
                 label='Historical')

    plt.plot(pd.to_datetime(X_forecast[timestamp_col]),
             y_forecast.values[:, var_id],
             c='#1E90FF',
             label='Forecast')

    if actual is not None:
        plt.plot(pd.to_datetime(X_test[timestamp_col]),
                 y_test.values[:, var_id],
                 c='#FF0000',
                 label='Actual')

    if forecast_interval is not None and show_forecast_interval:
        upper_forecast, lower_forecast = forecast_interval[0], forecast_interval[1]

        plt.fill_between(X_forecast[timestamp_col],
                         lower_forecast.values[:, var_id],
                         upper_forecast.values[:, var_id],
                         facecolor='#ADD8E6',
                         alpha=0.5,
                         label='Uncertainty Interval')

    plt.title('Actual vs Forecast' if actual is not None else 'Forecast Curve', fontsize=16)
    plt.ylabel(y_forecast.columns[var_id])
    plt.xlabel('Date')
    plt.grid(grid, alpha=0.3)
    plt.legend(fontsize=12, loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    plt.show()