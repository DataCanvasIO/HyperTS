# -*- coding:utf-8 -*-
"""

"""
from hypernets.searchers import make_searcher
from hypernets.discriminators import make_discriminator
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.tabular.cache import clear as _clear_cache
from hypernets.utils import logging, isnotebook, load_module

from hyperts.utils._base import get_tool_box
from hyperts.utils.metrics import metric_to_scorer
from hyperts.utils import consts, set_random_state
from hyperts.hyper_ts import HyperTS as hyper_ts_cls
from hyperts.framework.compete import TSCompeteExperiment

logger = logging.get_logger(__name__)


class HyperTS:
    """A class object for HyperTS.

    Parameters
    ----------
    train_data : str, Pandas or Dask or Cudf DataFrame.
        Feature data for training with target column.
        For str, it's should be the data path in file system, will be loaded as pnadas Dataframe.
        we'll detect data format from this path (only .csv and .parquet are supported now).
    task : str.
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
        See consts.py for details.
    eval_data : str, Pandas or Dask or Cudf DataFrame, optional.
        Feature data for evaluation, should be None or have the same python type with 'train_data'.
    test_data : str, Pandas or Dask or Cudf DataFrame, optional.
        Feature data for testing without target column, should be None or have the same python type with 'train_data'.
    mode : str, default 'stats'. Optional {'stats', 'dl', 'nas'}, where,
        'stats' indicates that all the models selected in the execution experiment are statistical models.
        'dl' indicates that all the models selected in the execution experiment are deep learning models.
        'nas' indicates that the selected model of the execution experiment will be a deep network model
        for neural architecture search, which is not currently supported.
    target : str, optional.
        Target feature name for training, which must be one of the train_data columns.
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
    """

    def __init__(self,
                 train_data,
                 task,
                 eval_data=None,
                 test_data=None,
                 mode='stats',
                 target=None,
                 freq=None,
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
                 log_level=None,
                 random_state=None,
                 clear_cache=None,
                 **kwargs):

        # 1. Check Data and Task
        assert train_data is not None, 'train data is required.'
        assert eval_data is None or type(eval_data) is type(train_data)
        assert test_data is None or type(test_data) is type(train_data)

        assert task is not None, 'task is required. Task naming paradigm:' \
                                 f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}'

        if task not in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
            ValueError(f'Task naming paradigm:'
                       f'{consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION}')

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

        self.task = task
        self.mode = mode
        self.target = target
        self.timestamp = timestamp
        self.freq = freq
        self.covariables = covariables

        self.pipeline_model = None
        self.excluded_variables = None
        self.history = None
        self.experiment = self.__build_experiment(train_data=train_data,
                                                  eval_data=eval_data,
                                                  test_data=test_data,
                                                  searcher=searcher,
                                                  search_space=search_space,
                                                  search_callbacks=search_callbacks,
                                                  searcher_options=searcher_options,
                                                  callbacks=callbacks,
                                                  id=id,
                                                  early_stopping_rounds=early_stopping_rounds,
                                                  early_stopping_time_limit=early_stopping_time_limit,
                                                  early_stopping_reward=early_stopping_reward,
                                                  reward_metric=reward_metric,
                                                  optimize_direction=optimize_direction,
                                                  discriminator=discriminator,
                                                  hyper_model_options=hyper_model_options,
                                                  log_level=log_level,
                                                  clear_cache=clear_cache,
                                                  random_state=random_state,
                                                  **kwargs)

    def __build_experiment(self, train_data, eval_data=None, test_data=None,
                           searcher=None, search_space=None, searcher_options=None,
                           callbacks=None, search_callbacks=None, id=None,
                           early_stopping_rounds=10, early_stopping_time_limit=3600,
                           early_stopping_reward=None, reward_metric=None,
                           optimize_direction=None, discriminator=None, hyper_model_options=None,
                           log_level=None, clear_cache=None, random_state=None, **kwargs):
        """Build a experiment object.

        Returns
        -------
        Runnable experiment object.
        """
        # 1. Load data
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

        # 2. Split X_train, y_train, X_eval, y_eval
        X_train, y_train, X_eval, y_eval = None, None, None, None
        if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
            if self.target is None:
                self.target = self._find_target(train_data)

            X_train, y_train = train_data.drop(columns=[self.target]), train_data.pop(self.target)

            if eval_data is not None:
                X_eval, y_eval = eval_data.drop(columns=[self.target]), eval_data.pop(self.target)
            if eval_data is not None:
                X_eval, y_eval = eval_data.drop(columns=[self.target]), eval_data.pop(self.target)

        elif self.task in consts.TASK_LIST_FORECAST:
            if self.covariables is not None:
                self.excluded_variables = [self.timestamp] + self.covariables
            else:
                self.excluded_variables = [self.timestamp]
            if self.target is None:
                self.target = tb.list_diff(train_data.columns.tolist(), self.excluded_variables)

            X_train, y_train = train_data[self.excluded_variables], train_data[self.target]
            if eval_data is not None:
                X_eval, y_eval = eval_data[self.excluded_variables], eval_data[self.target]

            self.freq = tb.infer_ts_freq(X_train, ts_name=self.timestamp)

        # 3. Task type infering
        if self.task == consts.Task_FORECAST and len(y_train.columns) == 1:
            self.task = consts.Task_UNIVARIATE_FORECAST
        elif self.task == consts.Task_FORECAST and len(y_train.columns) > 1:
            self.task = consts.Task_MULTIVARIATE_FORECAST

        if self.task == consts.Task_CLASSIFICATION:
            if y_train.nunique() == 2:
                if len(X_train.columns) == 1:
                    self.task = consts.Task_UNIVARIATE_BINARYCLASS
                else:
                    self.task = consts.Task_MULTIVARIATE_BINARYCLASS
            else:
                if len(X_train.columns) == 1:
                    self.task = consts.Task_UNIVARIATE_MULTICALSS
                else:
                    self.task = consts.Task_MULTIVARIATE_MULTICALSS
        logger.info(f'Inference could be type [{self.task}] task.')

        # 4. Configuration
        if reward_metric is None:
            if self.task in consts.TASK_LIST_FORECAST:
                reward_metric = 'mae'
            if self.task in consts.TASK_LIST_CLASSIFICATION:
                reward_metric = 'accuracy'
            if self.task in consts.TASK_LIST_REGRESSION:
                reward_metric = 'rmse'
            logger.info(f'no reward metric specified, use "{reward_metric}" for {self.task} task by default.')
        if isinstance(reward_metric, str):
            logger.info(f'Reward_metric is [{reward_metric}].')
        else:
            logger.info(f'Reward_metric is [{reward_metric.__name__}].')

        # 5. Get scorer
        if kwargs.get('scorer') is None:
            greater_is_better = kwargs.pop('greater_is_better', None)
            scorer = metric_to_scorer(reward_metric, task=self.task, pos_label=kwargs.get('pos_label'),
                                      greater_is_better=greater_is_better)
        else:
            scorer = kwargs.pop('scorer')
            if isinstance(scorer, str):
                raise ValueError('scorer should be a [make_scorer(metric, greater_is_better)] type.')

        # 6. Specify optimization direction
        if optimize_direction is None or len(optimize_direction) == 0:
            optimize_direction = 'max' if scorer._sign > 0 else 'min'
        logger.info(f'Optimize direction is [{optimize_direction}].')

        # 7. Get search space
        if (searcher is None or isinstance(searcher, str)) and search_space is None:
            search_space = self._default_search_space(self.task, self.mode, search_space,
                                                      timestamp=self.timestamp, metrics=reward_metric,
                                                      covariables=self.covariables)

        # 8. Get searcher
        searcher = self._to_search_object(search_space, searcher=searcher,
                                          optimize_direction=optimize_direction, searcher_options=searcher_options)
        logger.info(f'Searcher is [{searcher.__class__.__name__}].')

        # 9. Define callbacks
        if search_callbacks is None:
            search_callbacks = self._default_search_callbacks()
        search_callbacks = self._append_early_stopping_callbacks(search_callbacks, scorer=scorer,
                                                                 optimize_direction=optimize_direction,
                                                                 early_stopping_rounds=early_stopping_rounds,
                                                                 time_limit=early_stopping_time_limit,
                                                                 reward=early_stopping_reward)

        if callbacks is None:
            callbacks = self._default_experiment_callbacks()

        # 10. Define discriminator
        if discriminator is None and cfg.experiment_discriminator is not None and len(cfg.experiment_discriminator) > 0:
            discriminator = make_discriminator(cfg.experiment_discriminator, optimize_direction=optimize_direction,
                                               **(cfg.experiment_discriminator_options or {}))

        # 11. Define id
        if id is None:
            hasher = tb.data_hasher()
            id = hasher(dict(X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test,
                             eval_size=kwargs.get('eval_size'), target=self.target, task=self.task))
            id = f'{hyper_ts_cls.__name__}_{id}'

        # 12. Define hyper_model
        if hyper_model_options is None:
            hyper_model_options = {}
        hyper_model = hyper_ts_cls(searcher, mode=self.mode, reward_metric=reward_metric, task=self.task,
                          callbacks=search_callbacks, discriminator=discriminator, **hyper_model_options)

        # 13. Experiment
        experiment = TSCompeteExperiment(hyper_model, X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
                                         X_test=X_test,
                                         timestamp_col=self.timestamp, covariate_cols=self.covariables,
                                         log_level=log_level,
                                         task=self.task, id=id, callbacks=callbacks, scorer=scorer,
                                         random_state=random_state, **kwargs)
        if clear_cache:
            _clear_cache()

        if logger.is_info_enabled():
            train_shape = tb.get_shape(X_train)
            test_shape = tb.get_shape(X_test, allow_none=True)
            eval_shape = tb.get_shape(X_eval, allow_none=True)
            logger.info(f'make_experiment with train data:{train_shape}, '
                        f'test data:{test_shape}, eval data:{eval_shape}, target:{self.target}, task:{self.task}')

        self.history = {
            'train_data': train_data,
            'eval_data': eval_data
        }

        return experiment

    def fit(self, max_trials=3, **kwargs):
        """Runs the experiment.

        Parameters
        ----------
        max_trials: 'int' or None. Maximum number of experiments performed.
            Default 3.
        """

        model = self.experiment.run(max_trials=max_trials, **kwargs)
        self.pipeline_model = model
        return self

    def predict(self, X):
        """Predicts target for sequences in X.

        Parameters
        ----------
        X: 'DataFrame'.
            For forecast task, X.columns = ['timestamp', (covariate_1), (covariate_2),...].
            (covariate_1) indicates that it may not exist.
            For classification or regression tasks, X.columns = [variate_1, variate_2,...].
        """

        y_pred = self.pipeline_model.predict(X)

        if self.task in consts.TASK_LIST_FORECAST:
            tb = get_tool_box(X)
            X = tb.smooth_missed_ts_rows(X, ts_name=self.timestamp, freq=self.freq)
            forecast = tb.DataFrame(y_pred, columns=self.target)
            forecast = tb.concat_df([X, forecast], axis=1)
            return forecast
        else:
            return y_pred

    def predict_proba(self, X):
        """Predicts target probabilities for sequences in X for classification task.

        Parameters
        ----------
        X: 'DataFrame'.
            X.columns = [variate_1, variate_2,...].
        """

        if self.task in consts.TASK_LIST_CLASSIFICATION:
            y_proba = self.pipeline_model.predict_proba(X)
        else:
            raise ValueError('predict_proba is used for classification only.')
        return y_proba

    def evaluate(self, y_true, y_pred, y_proba=None, metrics=None):
        """Evaluates model performance.

        Parameters
        ----------
        y_true: 'np.arrray'.
        y_pred: 'pd.DataFrame' or 'np.arrray'.
            For forecast task, 'pd.DataFrame', X.columns could be ['timestamp', (covariate_1),
            (covariate_2),..., variate_1, variate_2,...].
            (covariate_1) indicates that it may not exist.
            For classification and regression tasks, 'np.arrray'.
        y_proba: 'np.arrray' or None, some metrics should be given, such as AUC.
        metrics: list, tuple or None. If None,
            For forecast or regression tasks, metrics = ['mae', 'mse', 'rmse', 'mape', 'smape'],
            For classification task, metrics = ['accuracy', 'f1', 'precision', 'recall'].
        """

        import pandas as pd
        from hyperts.utils.metrics import calc_score

        pd.set_option('display.max_columns', 10,
                      'display.max_rows', 10,
                      'display.float_format', lambda x: '%.4f' % x)

        if self.task in consts.TASK_LIST_FORECAST and metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape', 'smape']
        else:
            metrics = ['accuracy', 'f1', 'precision', 'recall']

        if self.task in consts.TASK_LIST_FORECAST:
            tb = get_tool_box(y_pred)
            y_pred = tb.df_to_array(y_pred[self.target])

        scores = calc_score(y_true, y_pred, y_proba=y_proba, metrics=metrics, task=self.task)

        scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        scores = scores.reset_index().rename(columns={'index': 'Metirc'})

        return scores

    def plot(self, forecast, actual=None, var_id=0, show_forecast_interval=True, include_history=True):
        """Plots forecast trend curves for the forecst task.

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
        actual: 'DataFrame' or None. If it is not None, the column needs to include
            the time column and the target column.
        var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
            it is the name of the target column. default 0.
        show_forecast_interval: 'bool'. Whether to show the forecast intervals.
            Default True.
        include_history: 'bool'. Whether to show the historical observations.
            Default True.

        Returns
        ----------
        fig : 'plotly.graph_objects.Figure'.
        """

        import plotly.graph_objects as go

        if isinstance(var_id, str) and var_id in self.target:
            var_id = self.target.index(var_id)
        elif isinstance(var_id, str) and var_id not in self.target:
            raise ValueError(f'{var_id} might not be target columns {self.target}.')

        X_forecast, y_forecast = self.split_X_y(forecast)

        if actual is not None:
            X_test, y_test = self.split_X_y(actual)
        else:
            X_test, y_test = None, None

        if include_history:
            if self.history['eval_data'] is not None:
                tb = get_tool_box(self.history['train_data'])
                data = tb.concat_df([self.history['train_data'], self.history['eval_data']], axis=0)
            else:
                data = self.history['train_data']
            X_train, y_train = self.split_X_y(data)
            train_end_date = X_train[self.timestamp].iloc[-1]
        else:
            X_train, y_train, train_end_date = None, None, None

        fig = go.Figure()

        if show_forecast_interval:
            if not include_history:
                X_train, y_train = self.split_X_y(self.history['train_data'])
            tb_y = get_tool_box(y_train)
            upper_forecast, lower_forecast = tb_y.infer_forecast_interval(y_train, y_forecast)

            lower_bound = go.Scatter(
                name='Lower Bound',
                x=X_forecast[self.timestamp],
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
                x=X_forecast['TimeStamp'],
                y=upper_forecast.values[:, 0],
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
                x=X_test[self.timestamp],
                y=y_test.values[:, var_id],
                mode='lines',
                line=dict(color='rgba(250, 43, 20, 0.7)'),
                name='Actual'
            )
            fig.add_trace(actual_trace)

        forecast_trace = go.Scatter(
            x=X_forecast[self.timestamp],
            y=y_forecast.values[:, var_id],
            mode='lines',
            line=dict(color='rgba(31, 119, 180, 0.7)'),
            name='Forecast'
        )
        fig.add_trace(forecast_trace)

        if train_end_date is not None:
            history_trace = go.Scatter(
                x=X_train[self.timestamp],
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
            yaxis=dict(title=y_forecast.columns[0]),
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

    def split_X_y(self, data, impute=False):
        """Splits the data into X and y.

        Parameters
        ----------
        data: 'DataFrame', including X and y.
        impute: Whether to impute in missing values. Default False.

        Returns
        -------
        X, y.
        """
        if self.task in consts.TASK_LIST_FORECAST:
            tb = get_tool_box(data)
            data = tb.drop_duplicated_ts_rows(data, ts_name=self.timestamp)
            data = tb.smooth_missed_ts_rows(data, ts_name=self.timestamp, freq=self.freq)

            if impute is not False:
                data[self.target] = tb.multi_period_loop_imputer(data[self.target], freq=self.freq)

            X, y = data[self.excluded_variables], data[self.target]
        else:
            X, y = data.drop(columns=[self.target]), data.pop(self.target)
        return X, y

    @staticmethod
    def _find_target(df):
        columns = df.columns.to_list()
        for col in columns:
            if col.lower() in cfg.experiment_default_target_set:
                return col
        raise ValueError(f'Not found one of {cfg.experiment_default_target_set} from your data,'
                         f' implicit target must be specified.')

    @staticmethod
    def _default_search_space(task, mode=consts.Mode_STATS, search_pace=None,
                              timestamp=None, covariables=None, metrics=None):
        if search_pace is not None:
            return search_pace

        if callable(metrics):
            metrics = [metrics.__name__]
        elif isinstance(metrics, str):
            metrics = [metrics.lower()]
        else:
            metrics = 'auto'

        if mode == consts.Mode_STATS and task in consts.TASK_LIST_FORECAST:
            from hyperts.macro_search_space import StatsForecastSearchSpace
            search_pace = StatsForecastSearchSpace(task=task, timestamp=timestamp,
                                                   covariables=covariables)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_CLASSIFICATION:
            from hyperts.macro_search_space import StatsClassificationSearchSpace
            search_pace = StatsClassificationSearchSpace(task=task, timestamp=timestamp)
        elif mode == consts.Mode_STATS and task in consts.TASK_LIST_REGRESSION:
            raise NotImplementedError(
                'STATSRegressionSearchSpace is not implemented yet.'
            )
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_FORECAST:
            from hyperts.macro_search_space import DLForecastSearchSpace
            search_pace = DLForecastSearchSpace(task=task, timestamp=timestamp,
                                                metrics=metrics, covariables=covariables)
        elif mode == consts.Mode_DL and task in consts.TASK_LIST_CLASSIFICATION:
            from hyperts.macro_search_space import DLClassificationSearchSpace
            search_pace = DLClassificationSearchSpace(task=task, timestamp=timestamp,
                                                      metrics=metrics)
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

    @staticmethod
    def _default_experiment_callbacks():
        cbs = cfg.experiment_callbacks_notebook if isnotebook() else cfg.experiment_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    @staticmethod
    def _default_search_callbacks():
        cbs = cfg.hyper_model_callbacks_notebook if isnotebook() else cfg.hyper_model_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    @staticmethod
    def _append_early_stopping_callbacks(cbs, scorer, optimize_direction='min',
                                         early_stopping_rounds=10, time_limit=3600, reward=None):
        from hypernets.core.callbacks import EarlyStoppingCallback

        assert isinstance(cbs, (tuple, list))
        if any([isinstance(cb, EarlyStoppingCallback) for cb in cbs]):
            return cbs

        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'
        es = EarlyStoppingCallback(early_stopping_rounds, op,
                                   time_limit=time_limit,
                                   expected_reward=reward)
        return [es] + cbs

    @staticmethod
    def _default_searcher(cls, search_space, optimize_direction='min', options=None):
        assert search_space is not None, '"search_space" should be specified when "searcher" is None or str.'
        assert optimize_direction in {'max', 'min'}
        if options is None:
            options = {}
        options['optimize_direction'] = optimize_direction
        s = make_searcher(cls, search_space, **options)
        return s

    def _to_search_object(self, search_space, searcher=None, optimize_direction='min', searcher_options=None):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher

        if searcher is None:
            searcher = self._default_searcher(EvolutionSearcher, search_space, optimize_direction, searcher_options)
        elif isinstance(searcher, (type, str)):
            searcher = self._default_searcher(searcher, search_space, optimize_direction, searcher_options)
        elif not isinstance(searcher, SearcherSpec):
            logger.warning(f'Unrecognized searcher "{searcher}".')

        return searcher