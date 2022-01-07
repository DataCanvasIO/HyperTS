# -*- coding:utf-8 -*-
"""

"""
import pandas as pd

from hypernets.searchers import make_searcher
from hypernets.discriminators import make_discriminator
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.utils import load_data, logging, isnotebook, load_module

from hyperts.utils._base import get_tool_box
from hyperts.utils.metrics import metric_to_scorer
from hyperts.utils import consts, tf_gpu, set_random_state
from hyperts.hyper_ts import HyperTS as hyper_ts_cls
from hyperts.framework.compete import TSCompeteExperiment
from hyperts.macro_search_space import (stats_forecast_search_space, stats_classification_search_space,
                                        dl_forecast_search_space, dl_classification_search_space)


logger = logging.get_logger(__name__)


def make_experiment(train_data,
                    task,
                    eval_data=None,
                    mode='stats',
                    target=None,
                    timestamp=None,
                    covariables=None,
                    id=None,
                    searcher=None,
                    search_space=None,
                    search_callbacks=None,
                    searcher_options=None,
                    callbacks=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    discriminator=None,
                    hyper_model_options=None,
                    dl_gpu_usage_strategy=0,
                    dl_memory_limit=2048,
                    log_level='info',
                    random_state=None,
                    **kwargs):
    """
    Parameters
    ----------
    train_data : str, Pandas or Dask or Cudf DataFrame.
        Feature data for training with target column.
        For str, it's should be the data path in file system, will be loaded as pnadas Dataframe.
        we'll detect data format from this path (only .csv and .parquet are supported now).

    task : str or None, (default=None)
        Task type(*binary*, *multiclass* or *regression*).
        If None, inference the type of task automatically

    eval_data : str, Pandas or Dask or Cudf DataFrame, optional
        Feature data for evaluation, should be None or have the same python type with 'train_data'.
    mode:

    target : str, optional
        Target feature name for training, which must be one of the drain_data columns, default is 'y'.

    id : str or None, (default=None)
        The experiment id.
    callbacks: list of ExperimentCallback, optional
        ExperimentCallback list.
    searcher : str, searcher class, search object, optional
        The hypernets Searcher instance to explore search space, default is EvolutionSearcher instance.
        For str, should be one of 'evolution', 'mcts', 'random'.
        For class, should be one of EvolutionSearcher, MCTSSearcher, RandomSearcher, or subclass of hypernets Searcher.
        For other, should be instance of hypernets Searcher.
    searcher_options: dict, optional, default is None
        The options to create searcher, is used if searcher is str.
    search_space : callable, optional
        Used to initialize searcher instance (if searcher is None, str or class).
    search_callbacks
        Hypernets search callbacks, used to initialize searcher instance (if searcher is None, str or class).
        If log_level >= WARNNING, default is EarlyStoppingCallback only.
        If log_level < WARNNING, defalult is EarlyStoppingCallback plus SummaryCallback.
    early_stopping_rounds :　int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 10.
    early_stopping_time_limit : int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 3600 seconds.
    early_stopping_reward : float, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is None.
    reward_metric : str, callable, optional, (default 'accuracy' for binary/multiclass task, 'rmse' for regression task)
        Hypernets search reward metric name or callable. Possible values:
            - accuracy
            - auc
            - f1
            - logloss
            - mse
            - mae
            - msle
            - precision
            - rmse
            - r2
            - recall
    optimize_direction : str, optional
        Hypernets search reward metric direction, default is detected from reward_metric.
    discriminator : instance of hypernets.discriminator.BaseDiscriminator, optional
        Discriminator is used to determine whether to continue training
    hyper_model_options: dict, optional
        Options to initlize HyperModel except *reward_metric*, *task*, *callbacks*, *discriminator*.
    evaluation_metrics: str, list, or None (default='auto'),
        If *eval_data* is not None, it used to evaluate model with the metrics.
        For str should be 'auto', it will selected metrics accord to machine learning task type.
        For list should be metrics name.
    evaluation_persist_prediction: bool (default=False)
    evaluation_persist_prediction_dir: str or None (default='predction')
        The dir to persist prediction, if exists will be overwritten
    report_render: str, obj, optional, default is None
        The experiment report render.
        For str should be one of 'excel'
        for obj should be instance ReportRender
    report_render_options: dict, optional
        The options to create render, is used if render is str.
    experiment_cls: class, or None, (default=CompeteExperiment)
        The experiment type, CompeteExperiment or it's subclass.
    clear_cache: bool, optional, (default False)
        Clear cache store before running the expeirment.
    log_level : int, str, or None, (default=None),
        Level of logging, possible values:
            -logging.CRITICAL
            -logging.FATAL
            -logging.ERROR
            -logging.WARNING
            -logging.WARN
            -logging.INFO
            -logging.DEBUG
            -logging.NOTSET

    kwargs:
        Parameters to initialize experiment instance, refrence TSCompeteExperiment for more details.
    Returns
    -------
    Runnable experiment object.

    """

    def find_target(df):
        columns = df.columns.to_list()
        for col in columns:
            if col.lower() in cfg.experiment_default_target_set:
                return col
        raise ValueError(f'Not found one of {cfg.experiment_default_target_set} from your data,'
                         f' implicit target must be specified.')

    def to_search_object(searcher, search_space):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher

        if searcher is None:
            searcher = default_searcher(EvolutionSearcher, search_space, searcher_options)
        elif isinstance(searcher, (type, str)):
            searcher = default_searcher(searcher, search_space, searcher_options)
        elif not isinstance(searcher, SearcherSpec):
            logger.warning(f'Unrecognized searcher "{searcher}".')

        return searcher

    def default_search_space(mode, task, search_pace=None, timestamp=None, metrics=None, covariables=None):
        if search_pace is not None:
            return search_pace
        if mode == consts.Mode_STATS and task in consts.TASK_LIST_FORECAST:
            search_pace = stats_forecast_search_space(task=task, timestamp=timestamp, covariables=covariables)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_CLASSIFICATION:
            search_pace = stats_classification_search_space(task=task, timestamp=timestamp)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_REGRESSION:
            search_pace = None
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_FORECAST:
            search_pace = dl_forecast_search_space(task=task, timestamp=timestamp, metrics=metrics, covariables=covariables)
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_CLASSIFICATION:
            search_pace = dl_classification_search_space(task=task, timestamp=timestamp, metrics=metrics)
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_REGRESSION:
            search_pace = None
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_FORECAST:
            search_pace = None
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_CLASSIFICATION:
            search_pace = None
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_REGRESSION:
            search_pace = None

        return search_pace

    def default_searcher(cls, search_space, options):
        assert search_space is not None, '"search_space" should be specified when "searcher" is None or str.'
        assert optimize_direction in {'max', 'min'}
        if options is None:
            options = {}
        options['optimize_direction'] = optimize_direction
        s = make_searcher(cls, search_space, **options)

        return s

    def default_experiment_callbacks():
        cbs = cfg.experiment_callbacks_notebook if isnotebook() else cfg.experiment_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    def default_search_callbacks():
        cbs = cfg.hyper_model_callbacks_notebook if isnotebook() else cfg.hyper_model_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    def append_early_stopping_callbacks(cbs):
        from hypernets.core.callbacks import EarlyStoppingCallback

        assert isinstance(cbs, (tuple, list))
        if any([isinstance(cb, EarlyStoppingCallback) for cb in cbs]):
            return cbs

        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'
        es = EarlyStoppingCallback(early_stopping_rounds, op,
                                   time_limit=early_stopping_time_limit,
                                   expected_reward=early_stopping_reward)
        return [es] + cbs

    # Parameters Checking
    assert train_data is not None, 'train data is required.'
    assert task is not None, 'task is required. Task naming paradigm:' \
                    f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}'

    if task not in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
        raise ValueError(f'Task naming paradigm:' 
                   f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}')

    kwargs = kwargs.copy()

    # Set Log Level
    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    # Set Random State
    if random_state is not None:
        set_random_state(seed=random_state, mode=mode)

    # Set GPU Usage Strategy for DL Mode
    if mode == consts.Mode_DL:
        if dl_gpu_usage_strategy == 0:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif dl_gpu_usage_strategy == 1:
            tf_gpu.set_memory_growth()
        elif dl_gpu_usage_strategy == 2:
            tf_gpu.set_memory_limit(limit=dl_memory_limit)
        else:
            raise ValueError(f'The GPU strategy is not supported. '
                             f'Default [0:cpu | 1:gpu-memory growth | 2: gpu-memory limit].')

    # Data Checking
    train_data, eval_data = [load_data(data) if data is not None else None for data in (train_data, eval_data)]

    tb = get_tool_box(train_data, eval_data)
    if hasattr(tb, 'is_dask_dataframe'):
        train_data, eval_data = [tb.reset_index(x) if tb.is_dask_dataframe(x) else x for x in (train_data, eval_data)]

    X_train, y_train, X_eval, y_eval = None, None, None, None
    if task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
        if target is None:
            target = find_target(train_data)
        X_train, y_train = train_data.drop(columns=[target]), train_data.pop(target)
        if eval_data is not None:
            X_eval, y_eval = eval_data.drop(columns=[target]), eval_data.pop(target)
        else:
            X_train, X_eval, y_train, y_eval = \
                tb.random_train_test_split(X_train, y_train, test_size=consts.DEFAULT_EVAL_SIZE)
    elif task in consts.TASK_LIST_FORECAST:
        excluded_variables = [timestamp] + covariables if covariables is not None else [timestamp]
        if target is None:
            target = tb.list_diff(train_data.columns.tolist(), excluded_variables)
        X_train, y_train = train_data[excluded_variables], train_data[target]
        if eval_data is not None:
            X_eval, y_eval = eval_data[excluded_variables], eval_data[target]
        else:
            X_train, X_eval, y_train, y_eval = \
                tb.temporal_train_test_split(X_train, y_train, test_size=consts.DEFAULT_EVAL_SIZE)

    # Task Type Infering
    if task == consts.Task_FORECAST and len(y_train.columns) == 1:
        task = consts.Task_UNIVARIATE_FORECAST
    elif task == consts.Task_FORECAST and len(y_train.columns) > 1:
        task = consts.Task_MULTIVARIATE_FORECAST

    if task == consts.Task_CLASSIFICATION:
        if y_train.nunique() == 2:
            if len(X_train.columns) == 1:
                task = consts.Task_UNIVARIATE_BINARYCLASS
            else:
                task = consts.Task_MULTIVARIATE_BINARYCLASS
        else:
            if len(X_train.columns) == 1:
                task = consts.Task_UNIVARIATE_MULTICALSS
            else:
                task = consts.Task_MULTIVARIATE_MULTICALSS
    logger.info(f'Inference task type could be [{task}].')

    # Configuration
    if reward_metric is None:
        if task in consts.TASK_LIST_FORECAST:
            reward_metric = 'mae'
        if task in consts.TASK_LIST_CLASSIFICATION:
            reward_metric = 'accuracy'
        if task in consts.TASK_LIST_REGRESSION:
            reward_metric = 'rmse'
        logger.info(f'No reward metric specified, use "{reward_metric}" for {task} task by default.')
    if isinstance(reward_metric, str):
        logger.info(f'Reward_metric is [{reward_metric}].')
    else:
        logger.info(f'Reward_metric is [{reward_metric.__name__}].')

    if kwargs.get('scorer') is None:
        greater_is_better = kwargs.pop('greater_is_better', None)
        scorer = metric_to_scorer(reward_metric, task=task, pos_label=kwargs.get('pos_label'),greater_is_better=greater_is_better)
    else:
        scorer = kwargs.pop('scorer')
        if isinstance(scorer, str):
            raise ValueError('scorer should be a [make_scorer(metric, greater_is_better)] type.')

    if optimize_direction is None or len(optimize_direction) == 0:
        optimize_direction = 'max' if scorer._sign > 0 else 'min'
    logger.info(f'Optimize direction is [{optimize_direction}].')

    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = default_search_space(mode, task, search_space, timestamp=timestamp, covariables=covariables)

    searcher = to_search_object(searcher, search_space)
    logger.info(f'Searcher is [{searcher.__class__.__name__}].')

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if callbacks is None:
        callbacks = default_experiment_callbacks()

    if discriminator is None and cfg.experiment_discriminator is not None and len(cfg.experiment_discriminator) > 0:
        discriminator = make_discriminator(cfg.experiment_discriminator,
                                           optimize_direction=optimize_direction,
                                           **(cfg.experiment_discriminator_options or {}))

    if id is None:
        hasher = tb.data_hasher()
        id = hasher(dict(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                         eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'{hyper_ts_cls.__name__}_{id}'

    if hyper_model_options is None:
        hyper_model_options = {}
    hyper_model = hyper_ts_cls(searcher, mode=mode, reward_metric=reward_metric, task=task, callbacks=search_callbacks,
                               discriminator=discriminator, **hyper_model_options)

    experiment = TSCompeteExperiment(hyper_model, X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                    timestamp_col=timestamp, covariate_cols=covariables, log_level=log_level,random_state=random_state,
                    task=task, id=id, callbacks=callbacks, scorer=scorer, **kwargs)

    return experiment


def process_test_data(test_df, timestamp=None, target=None, covariables=None, freq=None, impute=False):
    """
    Notes: timestamp is required for prediction tasks,
           target is required for classification and regression task.

    Parameters
    ----------


    Returns
    -------
          X_test, y_test.
    """

    tb = get_tool_box(test_df)

    if timestamp is not None:
        excluded_variables = [timestamp] + covariables if covariables is not None else [timestamp]
        if freq is None:
            freq = tb.infer_ts_freq(test_df[[timestamp]], ts_name=timestamp)
        if target is None:
            target = tb.list_diff(test_df.columns.tolist(), excluded_variables)
        test_df = tb.drop_duplicated_ts_rows(test_df, ts_name=timestamp)
        test_df = tb.smooth_missed_ts_rows(test_df, ts_name=timestamp, freq=freq)

        if impute is not False:
            test_df[target] = tb.multi_period_loop_imputer(test_df[target], freq=freq)

        X_test, y_test = test_df[excluded_variables], test_df[target]
        return X_test, y_test
    else:
        X_test = test_df
        y_test = X_test.pop(target)
        return X_test, y_test


def make_evaluation(y_true, y_pred, y_proba=None, task=None, metrics=None):
    from hyperts.utils.metrics import calc_score

    pd.set_option('display.max_columns', 10,
                  'display.max_rows', 10,
                  'display.float_format', lambda x: '%.4f' % x)

    if task in consts.TASK_LIST_FORECAST and metrics is None:
        metrics = ['r2', 'mae', 'mse', 'rmse', 'mape', 'smape']
    else:
        metrics = ['accuracy', 'f1', 'precision', 'recall']

    scores = calc_score(y_true, y_pred, y_proba=y_proba, metrics=metrics)

    scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    scores = scores.reset_index().rename(columns={'index': 'Metirc'})

    return scores


def forecast_plotly(test_df, y_pred, train_df=None, timestamp=None, covariables=None):
    import plotly.graph_objects as go

    X_test, y_test = process_test_data(test_df, timestamp=timestamp, covariables=covariables, impute=True)
    if train_df is not None:
        X_train, y_train = process_test_data(train_df, timestamp=timestamp, covariables=covariables, impute=True)
        train_end_date = X_train[timestamp].iloc[-1]
    else:
        X_train, y_train, train_end_date = None, None, None

    fig = go.Figure()

    forecast = go.Scatter(
        x=X_test[timestamp],
        y=y_pred.squeeze(),
        mode='lines',
        line=dict(color='rgba(250, 43, 20, 0.7)'),
        name='Forecast'
    )
    fig.add_trace(forecast)

    actual = go.Scatter(
        x=X_test[timestamp],
        y=y_test.values.squeeze(),
        mode='lines',
        line=dict(color='rgba(0, 90, 181, 0.8)'),
        name='Actual'
    )
    fig.add_trace(actual)

    if train_end_date is not None:
        train = go.Scatter(
            x=X_train[timestamp],
            y=y_train.squeeze(),
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.7)'),
            name='Historical'
        )
        fig.add_trace(train)

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
                    color="rgba(0, 90, 181, 0.7)",
                    width=1.0)
            )],

            annotations=[dict(
                xref="x",
                x=train_end_date,
                yref="paper",
                y=.95,
                text="Train End Date",
                showarrow=True,
                arrowhead=0,
                ax=-60,
                ay=0
            )]
        )
        fig.update_layout(new_layout)

    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title=y_test.columns[0]),
        title='Actual vs Forecast',
        title_x=0.5,
        showlegend=True,
        legend={'traceorder': 'reversed'},
        hovermode="x"
    )
    fig.update_layout(layout)

    fig.update_xaxes(rangeslider_visible=True)

    fig.show()