# -*- coding:utf-8 -*-
"""

"""
from hypernets.searchers import make_searcher
from hypernets.discriminators import make_discriminator
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.tabular.cache import clear as _clear_cache
from hypernets.utils import logging, isnotebook, load_module

from hyperts.utils import get_tool_box
from hyperts.utils import consts, set_random_state
from hyperts.hyper_ts import HyperTS as hyper_ts_cls
from hyperts.framework.compete import TSCompeteExperiment

logger = logging.get_logger(__name__)

def make_experiment(train_data,
                    task,
                    eval_data=None,
                    test_data=None,
                    mode='stats',
                    max_trials=3,
                    eval_size=0.2,
                    target=None,
                    freq=None,
                    timestamp=None,
                    timestamp_format='%Y-%m-%d %H:%M:%S',
                    covariables=None,
                    dl_forecast_window=None,
                    dl_forecast_horizon=1,
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
                    log_level=None,
                    random_state=None,
                    clear_cache=None,
                    **kwargs):
    """
    Parameters
    ----------
    train_data : str, Pandas or Dask or Cudf DataFrame.
        Feature data for training with target column.
        For str, it's should be the data path in file system, will be loaded as pnadas Dataframe.
        we'll detect data format from this path (only .csv and .parquet are supported now).
    task : str.
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass',
        'univariate-multiclass', 'multivariate-binaryclass, and ’multivariate-multiclass’.
        Notably, task can also configure 'forecast', 'classification', and 'regression'. At this point, HyprTS
        will perform detailed task type inference from the data combined with other known column information.
    eval_data : str, Pandas or Dask or Cudf DataFrame, optional.
        Feature data for evaluation, should be None or have the same python type with 'train_data'.
    test_data : str, Pandas or Dask or Cudf DataFrame, optional.
        Feature data for testing without target column, should be None or have the same python type with 'train_data'.
    max_trials : int, maximum number of tests (model search), optional, (default=3).
    eval_size : float or int, When the eval_data is None, customize the ratio to split the eval_data from
        the train_data. int indicates the prediction length of the forecast task. (default=0.2 or 10).
    mode : str, default 'stats'. Optional {'stats', 'dl', 'nas'}, where,
        'stats' indicates that all the models selected in the execution experiment are statistical models.
        'dl' indicates that all the models selected in the execution experiment are deep learning models.
        'nas' indicates that the selected model of the execution experiment will be a deep network model
        for neural architecture search, which is not currently supported.
    target : str or list, optional.
        Target feature name for training, which must be one of the train_data columns for classification[str],
        regression[str] or unvariate forecast task [list]. For multivariate forecast task, it is multiple columns
        of training data.
    freq: 'str', DateOffset or None, default None.
    timestamp : str, forecast task 'timestamp' cannot be None, (default=None).
    timestamp_format : str, the date format of timestamp col for forecast task, (default='%Y-%m-%d %H:%M:%S').
    covariables : list[n*str], if the data contains covariables, specify the covariable column names, (default=None).
    dl_forecast_window : int or None. When selecting 'dl' mode, you can specify window, which is the sequence
        length of each sample, (default=None).
    dl_forecast_horizon : int or None. When selecting 'dl' mode, you can specify horizon, which is the length of
        the interval between the input and the target, (default=1).
    id : str or None, (default=None).
        The experiment id.
    callbacks: list of ExperimentCallback, optional.
        ExperimentCallback list.
    searcher : str, searcher class, search object, optional.
        The hypernets Searcher instance to explore search space, default is EvolutionSearcher instance.
        For str, should be one of 'evolution', 'mcts', 'random'.
        For class, should be one of EvolutionSearcher, MCTSSearcher, RandomSearcher, or subclass of hypernets Searcher.
        For other, should be instance of hypernets Searcher.
    searcher_options: dict, optional, default is None.
        The options to create searcher, is used if searcher is str.
    search_space : callable, optional
        Used to initialize searcher instance (if searcher is None, str or class).
    search_callbacks
        Hypernets search callbacks, used to initialize searcher instance (if searcher is None, str or class).
        If log_level >= WARNNING, default is EarlyStoppingCallback only.
        If log_level < WARNNING, defalult is EarlyStoppingCallback plus SummaryCallback.
    early_stopping_rounds : int optional.
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 10.
    early_stopping_time_limit : int, optional.
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 3600 seconds.
    early_stopping_reward : float, optional.
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is None.
    reward_metric : str, callable, optional, (default 'accuracy' for binary/multiclass task, 'rmse' for
        forecast/regression task)
        Hypernets search reward metric name or callable. Possible values:
            - accuracy
            - auc
            - f1
            - logloss
            - mse
            - mae
            - rmse
            - mape
            - smape
            - msle
            - precision
            - r2
            - recall
    optimize_direction : str, optional.
        Hypernets search reward metric direction, default is detected from reward_metric.
    discriminator : instance of hypernets.discriminator.BaseDiscriminator, optional
        Discriminator is used to determine whether to continue training
    hyper_model_options: dict, optional.
        Options to initlize HyperModel except *reward_metric*, *task*, *callbacks*, *discriminator*.
    dl_gpu_usage_strategy : int, optional {0, 1, 2}.
        Deep neural net models(tensorflow) gpu usage strategy.
        0:cpu | 1:gpu-memory growth | 2: gpu-memory limit.
    dl_memory_limit : int, GPU memory limit, default 2048.
    random_state : int or None, default None.
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

    def default_search_space(task, search_space=None, metrics=None, covariables=None):
        if search_space is not None:
            return search_space

        if callable(metrics):
            metrics = [metrics.__name__]
        elif isinstance(metrics, str):
            metrics = [metrics.lower()]
        else:
            metrics = 'auto'

        if mode == consts.Mode_STATS and task in consts.TASK_LIST_FORECAST:
            from hyperts.macro_search_space import StatsForecastSearchSpace

            search_pace = StatsForecastSearchSpace(task=task, timestamp=timestamp, covariables=covariables)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_CLASSIFICATION:
            from hyperts.macro_search_space import StatsClassificationSearchSpace

            search_pace = StatsClassificationSearchSpace(task=task, timestamp=timestamp)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_REGRESSION:
            raise NotImplementedError(
                'STATSRegressionSearchSpace is not implemented yet.'
            )
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_FORECAST:
            from hyperts.macro_search_space import DLForecastSearchSpace

            search_pace = DLForecastSearchSpace(task=task, timestamp=timestamp, metrics=metrics, covariables=covariables,
                                                window=dl_forecast_window, horizon=dl_forecast_horizon)
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_CLASSIFICATION:
            from hyperts.macro_search_space import DLClassificationSearchSpace

            search_pace = DLClassificationSearchSpace(task=task, timestamp=timestamp, metrics=metrics)
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_REGRESSION:
            raise NotImplementedError(
                'DLRegressionSearchSpace is not implemented yet.'
            )
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_FORECAST:
            raise NotImplementedError(
                'NASForecastSearchSpace is not implemented yet.'
            )
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_CLASSIFICATION:
            raise NotImplementedError(
                'NASClassificationSearchSpace is not implemented yet.'
            )
        elif mode == consts.Mode_NAS and task in consts.TASK_LIST_REGRESSION:
            raise NotImplementedError(
                'NASRegressionSearchSpace is not implemented yet.'
            )
        else:
            raise ValueError('The default search space was not found!')

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

    # 1. Check Data and Task and Mode
    assert train_data is not None, 'train data is required.'
    assert eval_data is None or type(eval_data) is type(train_data)
    assert test_data is None or type(test_data) is type(train_data)

    assert task is not None, 'task is required. Task naming paradigm:' \
                    f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}'

    if task not in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
        raise ValueError(f'Task naming paradigm:' 
                   f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}')

    if task in consts.TASK_LIST_FORECAST and timestamp is None:
        raise ValueError("Forecast task 'timestamp' cannot be None.")

    if task in consts.TASK_LIST_FORECAST and covariables is None:
        logger.info('If the data contains covariables, specify the covariable column names.')

    if mode != consts.Mode_STATS:
        try:
            import tensorflow
        except ImportError:
            raise RuntimeError('Please install `tensorflow` package first. command: pip install tensorflow.')

    kwargs = kwargs.copy()
    kwargs['max_trials'] = max_trials
    kwargs['eval_size'] = eval_size

    # 2. Set Log Level
    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    # 3. Set Random State
    if random_state is not None:
        set_random_state(seed=random_state, mode=mode)

    # 4. Set GPU Usage Strategy for DL Mode
    if mode == consts.Mode_DL:
        if dl_gpu_usage_strategy == 0:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif dl_gpu_usage_strategy == 1:
            from hyperts.utils import tf_gpu
            tf_gpu.set_memory_growth()
        elif dl_gpu_usage_strategy == 2:
            from hyperts.utils import tf_gpu
            tf_gpu.set_memory_limit(limit=dl_memory_limit)
        else:
            raise ValueError(f'The GPU strategy is not supported. '
                             f'Default [0:cpu | 1:gpu-memory growth | 2: gpu-memory limit].')

    # 5. Load data
    if isinstance(train_data, str):
        import pandas as pd
        tb = get_tool_box(pd.DataFrame)
        train_data = tb.load_data(train_data, reset_index=True)
        eval_data = tb.load_data(eval_data, reset_index=True) if eval_data is not None else None
        X_test = tb.load_data(test_data, reset_index=True) if test_data is not None else None
    else:
        tb = get_tool_box(train_data, eval_data, test_data)
        train_data = tb.reset_index(train_data)
        eval_data = tb.reset_index(eval_data) if eval_data is not None else None
        X_test = tb.reset_index(test_data) if test_data is not None else None

    if task in consts.TASK_LIST_FORECAST:
        train_data[timestamp] = tb.datetime_format(train_data[timestamp], format=timestamp_format)
        if eval_data is not None:
            eval_data[timestamp] = tb.datetime_format(eval_data[timestamp], format=timestamp_format)
        if X_test is not None:
            X_test[timestamp] = tb.datetime_format(X_test[timestamp], format=timestamp_format)

    # 6. Split X_train, y_train, X_eval, y_eval
    X_train, y_train, X_eval, y_eval = None, None, None, None
    if task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
        if target is None:
            target = find_target(train_data)
        X_train, y_train = train_data.drop(columns=[target]), train_data.pop(target)
        if eval_data is not None:
            X_eval, y_eval = eval_data.drop(columns=[target]), eval_data.pop(target)
    elif task in consts.TASK_LIST_FORECAST:
        excluded_variables = [timestamp] + covariables if covariables is not None else [timestamp]
        if target is None:
            target = tb.list_diff(train_data.columns.tolist(), excluded_variables)
        elif target is not None and isinstance(target, str):
            target = [target]
        X_train, y_train = train_data[excluded_variables], train_data[target]
        if eval_data is not None:
            X_eval, y_eval = eval_data[excluded_variables], eval_data[target]

        if freq is None:
            freq = tb.infer_ts_freq(X_train, ts_name=timestamp)
            if freq is None:
                raise RuntimeError('Unable to infer correct frequency, please check data or specify frequency.')

    # 7. Covarite Transformer
    if covariables is not None:
        from hyperts.utils.transformers import CovariateTransformer
        cs = CovariateTransformer(covariables=covariables).fit(X_train)
        autual_covariables = cs.covariables_
    else:
        from hyperts.utils.transformers import IdentityTransformer
        cs = IdentityTransformer().fit(X_train)
        autual_covariables = covariables

    # 8. Infer Forecast Window for DL Mode
    if mode in [consts.Mode_DL, consts.Mode_NAS] and task in consts.TASK_LIST_FORECAST and dl_forecast_window is None:
        if eval_data is not None:
            max_win_size= int(len(X_eval) // 2 - dl_forecast_horizon)
        elif kwargs.get('eval_size') is not None:
            max_win_size = int(len(X_train)*kwargs['eval_size'] // 2 - dl_forecast_horizon)
        else:
            max_win_size = int(len(X_train)*consts.DEFAULT_EVAL_SIZE // 2 - dl_forecast_horizon)

        if max_win_size < 1:
            logger.warning('The trian data is too short to start dl mode, '
                           'stats mode has been automatically switched.')
            mode = consts.Mode_STATS
        else:
            dl_forecast_window = tb.infer_window_size(max_size=max_win_size, freq=freq)

    # 9. Task Type Infering
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

    # 10. Configuration
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

    # 11. Get scorer
    if kwargs.get('scorer') is None:
        scorer = tb.metrics.metric_to_scorer(reward_metric, task=task, pos_label=kwargs.get('pos_label'),
                                                                  optimize_direction=optimize_direction)
    else:
        scorer = kwargs.pop('scorer')
        if isinstance(scorer, str):
            raise ValueError('scorer should be a [make_scorer(metric, greater_is_better)] type.')

    # 12. Specify optimization direction
    if optimize_direction is None or len(optimize_direction) == 0:
        optimize_direction = 'max' if scorer._sign > 0 else 'min'
    logger.info(f'Optimize direction is [{optimize_direction}].')

    # 13. Get search space
    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = default_search_space(task=task, search_space=search_space,
                           metrics=reward_metric, covariables=autual_covariables)

    # 14. Get searcher
    searcher = to_search_object(searcher, search_space)
    logger.info(f'Searcher is [{searcher.__class__.__name__}].')

    # 15. Define callbacks
    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if callbacks is None:
        callbacks = default_experiment_callbacks()

    # 16. Define discriminator
    if discriminator is None and cfg.experiment_discriminator is not None and len(cfg.experiment_discriminator) > 0:
        discriminator = make_discriminator(cfg.experiment_discriminator,
                                           optimize_direction=optimize_direction,
                                           **(cfg.experiment_discriminator_options or {}))
    # 17. Define id
    if id is None:
        hasher = tb.data_hasher()
        id = hasher(dict(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                         eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'{hyper_ts_cls.__name__}_{id}'

    # 18. Define hyper_model
    if hyper_model_options is None:
        hyper_model_options = {}
    hyper_model = hyper_ts_cls(searcher, mode=mode, reward_metric=reward_metric, task=task,
            callbacks=search_callbacks, discriminator=discriminator, **hyper_model_options)

    # 19. Build Experiment
    experiment = TSCompeteExperiment(hyper_model, X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                                     task=task, mode=mode, timestamp_col=timestamp, target_col=target,
                                     covariate_cols=[covariables, autual_covariables], covariate_cleaner=cs,
                                     freq=freq, log_level=log_level, random_state=random_state,
                                     optimize_direction=optimize_direction, scorer=scorer,
                                     id=id, callbacks=callbacks, **kwargs)

    # 20. Clear Cache
    if clear_cache:
        _clear_cache()

    if logger.is_info_enabled():
        train_shape = tb.get_shape(X_train)
        test_shape = tb.get_shape(X_test, allow_none=True)
        eval_shape = tb.get_shape(X_eval, allow_none=True)
        logger.info(f'make_experiment with train data:{train_shape}, '
                    f'test data:{test_shape}, eval data:{eval_shape}, target:{target}, task:{task}')

    return experiment