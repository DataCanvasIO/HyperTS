# -*- coding:utf-8 -*-
"""

"""
from hypernets.searchers import make_searcher
from hypernets.discriminators import make_discriminator
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.utils import load_data, logging, isnotebook, load_module

from hyperts.utils import consts
from hyperts.utils.metrics import metric_to_scorer
from hyperts.hyper_ts import HyperTS as hyper_ts_cls
from hyperts.framework.compete import TSCompeteExperiment
from hyperts.macro_search_space import stats_forecast_search_space, stats_classification_search_space
from hyperts.utils._base import get_tool_box

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
                    log_level='info',
                    **kwargs):
    """
    Parameters
    ----------

    kwargs:
        Parameters to initialize experiment instance, refrence CompeteExperiment for more details.
    Returns
    -------
    Runnable experiment object

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

    def default_search_space(mode, task, search_pace, timestamp=None, covariables=None):
        if search_pace is not None:
            return search_pace
        if mode == consts.Mode_STATS and task in consts.TASK_LIST_FORECAST:
            search_pace = stats_forecast_search_space(task=task, timestamp=timestamp, covariables=covariables)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_CLASSIFICATION:
            search_pace = stats_classification_search_space(task=task, timestamp=timestamp)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_REGRESSION:
            search_pace = None
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_FORECAST:
            search_pace = None
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_CLASSIFICATION:
            search_pace = None
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

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

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
    hyper_model = hyper_ts_cls(searcher, reward_metric=reward_metric, task=task, callbacks=search_callbacks,
                               discriminator=discriminator, **hyper_model_options)

    experiment = TSCompeteExperiment(hyper_model, X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                                     timestamp_col=timestamp, covariate_cols=covariables, log_level=log_level,
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