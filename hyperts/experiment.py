# -*- coding:utf-8 -*-
"""

"""
from sklearn.metrics import get_scorer

from hypernets.tabular import get_tool_box
from hypernets.searchers import make_searcher
from hypernets.discriminators import make_discriminator
from hypernets.tabular.metrics import metric_to_scoring
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.utils import load_data, logging, isnotebook, load_module

from hyperts.utils import consts, toolbox as dp
from hyperts.hyper_ts import HyperTS as hyper_ts_cls
from hyperts.framework.compete import TSCompeteExperiment
from hyperts.macro_search_space import stats_forecast_search_space, stats_classification_search_space

logger = logging.get_logger(__name__)

forecast_task_list = [
    consts.Task_UNIVARIABLE_FORECAST,
    consts.Task_MULTIVARIABLE_FORECAST,
    consts.Task_FORECAST
]

classfication_task_list = [
    consts.Task_BINARY_CLASSIFICATION,
    consts.Task_MULTICLASS_CLASSIFICATION,
    consts.Task_CLASSIFICATION
]

regression_task_list = [
    consts.Task_REGRESSION
]


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

    def default_search_space(mode, task, timestamp=None, covariables=None):
        if mode == consts.Mode_STATS and task in forecast_task_list:
            search_pace = stats_forecast_search_space(task=task, timestamp=timestamp, covariables=covariables)
        elif mode == consts.Mode_STATS and task in classfication_task_list:
            search_pace = stats_classification_search_space(task=task, timestamp=timestamp)
        elif mode == consts.Mode_STATS and task in regression_task_list:
            search_pace = None
        elif mode == consts.Mode_DL and task in forecast_task_list:
            search_pace = None
        elif mode == consts.Mode_DL and task in classfication_task_list:
            search_pace = None
        elif mode == consts.Mode_DL and task in regression_task_list:
            search_pace = None
        elif mode == consts.Mode_NAS and task in forecast_task_list:
            search_pace = None
        elif mode == consts.Mode_NAS and task in classfication_task_list:
            search_pace = None
        elif mode == consts.Mode_NAS and task in regression_task_list:
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

    # Parameters checking
    assert train_data is not None, 'train data is required.'
    assert task is not None, 'task is required. Task naming paradigm:' \
                             f'{forecast_task_list + classfication_task_list + regression_task_list}'

    if task not in [forecast_task_list + classfication_task_list + regression_task_list]:
        ValueError(f'Task naming paradigm:' 
                   f'{forecast_task_list + classfication_task_list + regression_task_list}')

    kwargs = kwargs.copy()

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    # Data checking
    train_data, eval_data = [load_data(data) if data is not None else None for data in (train_data, eval_data)]

    tb = get_tool_box(train_data, eval_data)
    if hasattr(tb, 'is_dask_dataframe'):
        train_data, eval_data = [tb.reset_index(x) if tb.is_dask_dataframe(x) else x for x in (train_data, eval_data)]

    X_train, y_train, X_eval, y_eval = None, None, None, None
    if task in classfication_task_list + regression_task_list:
        if target is None:
            target = find_target(train_data)
        X_train, y_train = train_data.drop(columns=[target]), train_data.pop(target)
        if eval_data is not None:
            X_eval, y_eval = eval_data.drop(columns=[target]), eval_data.pop(target)
        else:
            X_train, X_eval, y_train, y_eval = \
                dp.random_train_test_split(X_train, y_train, test_size=0.1)
    elif task in forecast_task_list:
        if target is None:
            target = dp.list_diff(train_data.columns.tolist(), [timestamp] + covariables)
        X_train, y_train = train_data[[timestamp] + covariables], train_data[target]
        if eval_data is not None:
            X_eval, y_eval = eval_data[[timestamp] + covariables], eval_data[target]
        else:
            X_eval, y_eval = None, None

    if task == consts.Task_FORECAST and len(y_train.columns) == 1:
        task = consts.Task_UNIVARIABLE_FORECAST
    if task == consts.Task_CLASSIFICATION and y_train.nunique() == 2:
        task = consts.Task_BINARY_CLASSIFICATION

    if reward_metric is None:
        if task in forecast_task_list:
            reward_metric = 'mae'
        if task in classfication_task_list:
            reward_metric = 'accuracy'
        if task in regression_task_list:
            reward_metric = 'rmse'
        logger.info(f'no reward metric specified, use "{reward_metric}" for {task} task by default.')

    if kwargs.get('scorer') is None:
        scorer = metric_to_scoring(reward_metric, task=task, pos_label=kwargs.get('pos_label'))
    else:
        scorer = kwargs.pop('scorer')

    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    if optimize_direction is None or len(optimize_direction) == 0:
        optimize_direction = 'max' if scorer._sign > 0 else 'min'

    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = default_search_space(mode, task, timestamp=timestamp, covariables=covariables)

    searcher = to_search_object(searcher, search_space)

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
                                     timestamp_col=timestamp, covariate_cols=covariables,
                                     task=task, id=id, callbacks=callbacks, scorer=scorer, **kwargs)

    return experiment


def process_test_data(test_df, timestamp, covariables, freq=None, impute=False):
    if freq is None:
        freq = dp.infer_ts_freq(test_df[[timestamp]])
    target_varibales = dp.list_diff(test_df.columns.tolist(), [timestamp] + covariables)
    test_df = dp.drop_duplicated_ts_rows(test_df, ts_name=timestamp)
    test_df = dp.smooth_missed_ts_rows(test_df, ts_name=timestamp, freq=freq)

    if impute is not False:
        test_df[target_varibales] = dp.multi_period_loop_imputer(test_df[target_varibales], freq=freq)

    X_test, y_test = test_df[[timestamp] + covariables], test_df[target_varibales]
    return X_test, y_test