# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from hyperts.config import Config as cfg
from hyperts.utils import consts
from hyperts.utils.transformers import TimeSeriesHyperTransformer

from hyperts.framework.search_space import SearchSpaceMixin, WithinColumnSelector
from hyperts.framework.estimators import ProphetForecastEstimator
from hyperts.framework.estimators import ARIMAForecastEstimator
from hyperts.framework.estimators import VARForecastEstimator
from hyperts.framework.estimators import TSFClassificationEstimator
from hyperts.framework.estimators import KNNClassificationEstimator
from hyperts.framework.estimators import DeepARForecastEstimator
from hyperts.framework.estimators import HybirdRNNGeneralEstimator
from hyperts.framework.estimators import LSTNetGeneralEstimator
from hyperts.framework.estimators import NBeatsForecastEstimator
from hyperts.framework.estimators import InceptionTimeGeneralEstimator
from hyperts.framework.estimators import IForestDetectionEstimator
from hyperts.framework.estimators import OCSVMDetectionEstimator
from hyperts.framework.estimators import ConvVAEDetectionEstimator

from hypernets.tabular import column_selector as tcs
from hypernets.core.ops import HyperInput, ModuleChoice, Optional
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.pipeline.transformers import SimpleImputer
from hypernets.pipeline.transformers import StandardScaler
from hypernets.pipeline.transformers import MinMaxScaler
from hypernets.pipeline.transformers import MaxAbsScaler
from hypernets.pipeline.transformers import SafeOrdinalEncoder
from hypernets.pipeline.transformers import AsTypeTransformer

from hypernets.pipeline.base import Pipeline, DataFrameMapper
from hypernets.utils import logging, get_params

logger = logging.get_logger(__name__)


##################################### Define Data Proprecessing Pipeline #####################################
def categorical_transform_pipeline(covariables=None, impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    steps = [
        AsTypeTransformer(dtype='str', name=f'categorical_as_object_{seq_no}'),
        SimpleImputer(missing_values=np.nan,
                      strategy=impute_strategy,
                      name=f'categorical_imputer_{seq_no}'),
        SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}',
                           dtype='int32')
    ]
    if covariables is not None:
        cs = WithinColumnSelector(tcs.column_object_category_bool, covariables)
    else:
        cs = tcs.column_object_category_bool
    pipeline = Pipeline(steps, columns=cs,
                        name=f'categorical_covariable_transform_pipeline_{seq_no}')
    return pipeline


def numeric_transform_pipeline(covariables=None, impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)

    imputer = SimpleImputer(missing_values=np.nan,
                            strategy=impute_strategy,
                            name=f'numeric_imputer_{seq_no}',
                            force_output_as_float=True)
    scaler_options = ModuleChoice(
        [
            StandardScaler(name=f'numeric_standard_scaler_{seq_no}'),
            MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}'),
            MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}')
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')
    if covariables == None:
        cs = WithinColumnSelector(tcs.column_number_exclude_timedelta, covariables)
    else:
        cs = tcs.column_number_exclude_timedelta
    pipeline = Pipeline([imputer, scaler_optional], columns=cs,
                        name=f'numeric_covariate_transform_pipeline_{seq_no}')
    return pipeline


##################################### Define Base Search Space Generator #####################################

class _HyperEstimatorCreator:

    def __init__(self, cls, init_kwargs, fit_kwargs):
        super(_HyperEstimatorCreator, self).__init__()

        self.estimator_cls = cls
        self.estimator_fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.estimator_init_kwargs = init_kwargs if init_kwargs is not None else {}

    def __call__(self, *args, **kwargs):
        return self.estimator_cls(self.estimator_fit_kwargs, **self.estimator_init_kwargs)


class BaseSearchSpaceGenerator:

    def __init__(self, task, **kwargs) -> None:
        super().__init__()
        self.task = task
        self.options = kwargs

    @property
    def estimators(self):
        raise NotImplementedError('Please define estimators in here.')

    def create_preprocessor(self, hyper_input, options):
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)
        covariables = options.pop('covariables', self.covariables)
        timestamp = options.pop('timestamp', self.timestamp)
        pipelines = []

        if covariables is not None:
            # category
            if cfg.category_pipeline_enabled:
                pipelines.append(categorical_transform_pipeline(covariables=covariables)(hyper_input))
            # numeric
            if cfg.numeric_pipeline_enabled:
                pipelines.append(numeric_transform_pipeline(covariables=covariables)(hyper_input))
        # timestamp
        if timestamp is not None:
            pipelines.append(Pipeline([TimeSeriesHyperTransformer()],
                                      columns=[timestamp],
                                      name=f'timestamp_transform_pipeline_0')(hyper_input))

        preprocessor = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                       df_out_dtype_transforms=[(tcs.column_object, 'int')])(pipelines)
        return preprocessor

    def create_estimators(self, hyper_input, options):
        assert len(self.estimators.keys()) > 0

        creators = [_HyperEstimatorCreator(pairs[0],
                                           init_kwargs=self._merge_dict(pairs[1],
                                                                        options.pop(f'{k}_init_kwargs', None)),
                                           fit_kwargs=self._merge_dict(pairs[2], options.pop(f'{k}_fit_kwargs', None)))
                    for k, pairs in self.estimators.items()]

        unused = {}
        for k, v in options.items():
            used = False
            for c in creators:
                if k in c.estimator_init_kwargs.keys():
                    c.estimator_init_kwargs[k] = v
                    used = True
                if k in c.estimator_fit_kwargs.keys():
                    used = True
            if not used:
                unused[k] = v
        if len(unused) > 0:
            for c in creators:
                c.estimator_fit_kwargs.update(unused)

        estimators = [c() for c in creators]
        return ModuleChoice(estimators, name='estimator_options')(hyper_input)

    def __call__(self, *args, **kwargs):
        options = self._merge_dict(self.options, kwargs)

        space = HyperSpace()
        with space.as_default():
            hyper_input = HyperInput(name='input1')
            if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
                self.create_estimators(hyper_input, options)
            elif self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
                self.create_estimators(self.create_preprocessor(hyper_input, options), options)
            space.set_inputs(hyper_input)

        return space

    def _merge_dict(self, *args):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
        return d

    def __repr__(self):
        params = get_params(self)
        params.update(self.options)
        repr_ = ', '.join(['%s=%r' % (k, v) for k, v in params.items()])
        return f'{type(self).__name__}({repr_})'


##################################### Define Specific Search Space Generator #####################################

class StatsForecastSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Statistical Search Space for Time Series Forecasting.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'univariate-forecast' or
        'multivariate-forecast'.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support mse, mae, rmse, mape, smape, msle, and so on.
    enable_prophet: bool, default True.
    enable_arima: bool, default True.
    enable_var: bool, default True.
    prophet_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which prophet is searched.
    arima_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which arima is searched.
    var_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which var is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of deepar_init_kwargs, hybirdrnn_init_kwargs and lstnet_init_kwargs,
        you can refer to `hyperts.framework.estimators.ProphetForecastEstimator,
        hyperts.framework.estimators.ARIMAForecastEstimator, and
        hyperts.framework.estimators.VARForecastEstimator.`
    2. If other parameters exist, set them directly. For example, covariables=['is_holiday'].
    """
    def __init__(self, task=None, timestamp=None,
                 enable_prophet=True,
                 enable_arima=True,
                 enable_var=True,
                 prophet_init_kwargs=None,
                 arima_init_kwargs=None,
                 var_init_kwargs=None,
                 drop_observed_sample=True,
                 **kwargs):
        if enable_prophet and prophet_init_kwargs is not None:
            kwargs['prophet_init_kwargs'] = prophet_init_kwargs
        if enable_arima and arima_init_kwargs is not None:
            kwargs['arima_init_kwargs'] = arima_init_kwargs
        if enable_var and var_init_kwargs is not None:
            kwargs['var_init_kwargs'] = var_init_kwargs
        super(StatsForecastSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.enable_prophet = enable_prophet
        self.enable_arima = enable_arima
        self.enable_var = enable_var
        self.drop_observed_sample = drop_observed_sample

    @property
    def default_prophet_init_kwargs(self):
        default_init_kwargs = {
            'freq': self.freq,

            'seasonality_mode': Choice(['additive', 'multiplicative']),
            'changepoint_prior_scale': Choice([0.001, 0.01, 0.1, 0.5]),
            'seasonality_prior_scale': Choice([0.01, 0.1, 1.0, 10.0]),
            'holidays_prior_scale': Choice([0.01, 0.1, 1.0, 10.0]),
            'changepoint_range': Choice([0.8, 0.85, 0.9, 0.95]),

            # 'y_scale': Choice(['none-scale', 'min_max', 'max_abs', 'z_scale']),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_prophet_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def default_arima_init_kwargs(self):
        default_init_kwargs = {
            'freq': self.freq,

            'p': Choice([1, 2, 3, 4, 5]),
            'd': Choice([0, 1, 2]),
            'q': Choice([0, 1, 2]),
            'trend': Choice(['n', 'c', 't']),
            'seasonal_order': Choice([(1, 0, 0), (1, 0, 1), (1, 1, 1),
                                      (0, 1, 1), (1, 1, 0), (0, 1, 0)]),
            # 'period_offset': Choice([0, 0, 0, 0, 0, 0, 1, -1, 2, -2]),

            'y_scale': Choice(['none-scale', 'min_max', 'max_abs', 'z_scale']),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_arima_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def default_var_init_kwargs(self):
        default_init_kwargs = {
            # 'ic': Choice(['aic', 'fpe', 'hqic', 'bic']),
            'maxlags': Choice([None, 2, 6, 12, 24, 48]),
            'trend': Choice(['c', 'ct', 'ctt', 'nc', 'n']),
            'y_log': Choice(['none-log']*4+['logx']*1),
            'y_scale': Choice(['min_max']*5+['z_scale']*2+['max_abs']*1)
        }

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_var_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_prophet and ProphetForecastEstimator().is_prophet_installed:
            univar_containers['prophet'] = (
            ProphetForecastEstimator, self.default_prophet_init_kwargs, self.default_prophet_fit_kwargs)
        if self.enable_arima:
            univar_containers['arima'] = (
            ARIMAForecastEstimator, self.default_arima_init_kwargs, self.default_arima_fit_kwargs)
        if self.enable_var:
            multivar_containers['var'] = (
            VARForecastEstimator, self.default_var_init_kwargs, self.default_var_fit_kwargs)

        if self.task == consts.Task_UNIVARIATE_FORECAST:
            return univar_containers
        elif self.task == consts.Task_MULTIVARIATE_FORECAST:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIATE_FORECAST}'
                             f' or {consts.Task_MULTIVARIATE_FORECAST}.')


class StatsClassificationSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Statistical Search Space for Time Series Classification.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'univariate-binaryclass',
        'univariate-multiclass', 'multivariate-binaryclass, or ’multivariate-multiclass’.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support accuracy, f1, auc, recall, precision.
    enable_tsf: bool, default True.
    enable_knn: bool, default True.
    tsf_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which tsf is searched.
    knn_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which knn is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of tsf_init_kwargs, knn_init_kwargs,
        you can refer to `hyperts.framework.estimators.TSFClassificationEstimator, and
        hyperts.framework.estimators.KNNClassificationEstimator.`
    2. If other parameters exist, set them directly. For example, n_estimators=200.
    """
    def __init__(self, task=None, timestamp=None,
                 enable_tsf=True,
                 enable_knn=True,
                 tsf_init_kwargs=None,
                 knn_init_kwargs=None,
                 **kwargs):
        if hasattr(kwargs, 'covariables'):
            kwargs.pop('covariables', None)
        if enable_tsf and tsf_init_kwargs is not None:
            kwargs['tsf_init_kwargs'] = tsf_init_kwargs
        if enable_knn and knn_init_kwargs is not None:
            kwargs['knn_init_kwargs'] = knn_init_kwargs
        super(StatsClassificationSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.enable_tsf = enable_tsf
        self.enable_knn = enable_knn

    @property
    def default_tsf_init_kwargs(self):
        return {
            'min_interval': Choice([3, 5, 7]),
            'n_estimators': Choice([50, 100, 200, 300, 500]),
        }

    @property
    def default_tsf_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def default_knn_univar_init_kwargs(self):
        return {
            'n_neighbors': Choice([1, 3, 5, 7, 9, 15]),
            'weights': Choice(['uniform', 'distance']),
            'distance': Choice(['dtw', 'ddtw', 'lcss']),
            'x_scale': Choice(['z_score', 'scale-none'])
        }

    @property
    def default_knn_multivar_init_kwargs(self):
        return {
            'n_neighbors': Choice([1, 3, 5, 7, 9, 15]),
            'weights': Choice(['uniform', 'distance']),
            'distance': Choice(['dtw', 'ddtw', 'lcss']),
            'x_scale': Choice(['z_score', 'scale-none'])
        }

    @property
    def default_knn_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_tsf:
            univar_containers['tsf'] = (
            TSFClassificationEstimator, self.default_tsf_init_kwargs, self.default_tsf_fit_kwargs)
        if self.enable_knn:
            univar_containers['knn'] = (
            KNNClassificationEstimator, self.default_knn_univar_init_kwargs, self.default_knn_fit_kwargs)
            multivar_containers['knn'] = (
            KNNClassificationEstimator, self.default_knn_multivar_init_kwargs, self.default_knn_fit_kwargs)

        if self.task in [consts.Task_UNIVARIATE_BINARYCLASS, consts.Task_UNIVARIATE_MULTICALSS]:
            return univar_containers
        elif self.task in [consts.Task_MULTIVARIATE_BINARYCLASS, consts.Task_MULTIVARIATE_MULTICALSS]:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIATE_BINARYCLASS}'
                             f', {consts.Task_UNIVARIATE_MULTICALSS}, {consts.Task_MULTIVARIATE_BINARYCLASS}'
                             f', or {consts.Task_MULTIVARIATE_MULTICALSS}.')


class DLForecastSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Deep Learning Search Space for Time Series Forecasting.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'univariate-forecast' or
        'multivariate-forecast'.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support mse, mae, rmse, mape, smape, msle, and so on.
    enable_deepar: bool, default True.
    enable_hybirdrnn: bool, default True.
    enable_lstnet: bool, default True.
    deepar_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which deepar is searched.
    hybirdrnn_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which hybirdrnn is searched.
    lstnet_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which lstnet is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of deepar_init_kwargs, hybirdrnn_init_kwargs and lstnet_init_kwargs,
        you can refer to `hyperts.framework.estimators.DeepARForecastEstimator,
        hyperts.framework.estimators.HybirdRNNGeneralEstimator, and
        hyperts.framework.estimators.LSTNetGeneralEstimator.`
    2. If other parameters exist, set them directly. For example, covariables=['is_holiday'].
    """
    def __init__(self, task=None, timestamp=None, metrics=None,
                 window=None, horizon=1,
                 enable_deepar=True,
                 enable_hybirdrnn=True,
                 enable_lstnet=True,
                 enable_nbeats=True,
                 deepar_init_kwargs=None,
                 hybirdrnn_init_kwargs=None,
                 lstnet_init_kwargs=None,
                 nbeats_init_kwargs=None,
                 drop_observed_sample=True,
                 **kwargs):
        if enable_deepar and deepar_init_kwargs is not None:
            kwargs['deepar_init_kwargs'] = deepar_init_kwargs
        if enable_hybirdrnn and hybirdrnn_init_kwargs is not None:
            kwargs['hybirdrnn_init_kwargs'] = hybirdrnn_init_kwargs
        if enable_lstnet and lstnet_init_kwargs is not None:
            kwargs['lstnet_init_kwargs'] = lstnet_init_kwargs
        if enable_nbeats and nbeats_init_kwargs is not None:
            kwargs['nbeats_init_kwargs'] = nbeats_init_kwargs
        super(DLForecastSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.metrics = metrics
        self.window = window
        self.horizon = horizon
        self.enable_deepar = enable_deepar
        self.enable_hybirdrnn = enable_hybirdrnn
        self.enable_lstnet = enable_lstnet
        self.enable_nbeats = enable_nbeats
        self.drop_observed_sample = drop_observed_sample

    @property
    def default_deepar_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'horizon': self.horizon,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'optimizer': 'adam',
            'loss': 'log_gaussian_loss',
            'rnn_type': Choice(['gru', 'lstm']),
            'rnn_units': Choice([64]*2+[128]*3+[256]*2),
            'rnn_layers': Choice([2, 3]),
            'drop_rate': Choice([0., 0.1, 0.2]),
            'forecast_length': Choice([1]*8+[3, 6]),
            'window': Choice(self.window if isinstance(self.window, list) else [self.window]),

            'y_log': Choice(['none-log']*4+['logx']*1),
            'y_scale': Choice(['min_max']*4+['z_scale']*1),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_deepar_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def default_hybirdrnn_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'optimizer': 'adam',
            'loss': Choice(['mae', 'mse', 'huber_loss']),
            'rnn_type': Choice(['gru', 'lstm']),
            'rnn_units': Choice([64]*2+[128]*3+[256]*2),
            'rnn_layers': Choice([2, 3]),
            'drop_rate': Choice([0., 0.1, 0.2]),
            'forecast_length': Choice([1]*8+[3, 6]),
            'window': Choice(self.window if isinstance(self.window, list) else [self.window]),

            'y_log': Choice(['none-log']*4+['logx']*1),
            'y_scale': Choice(['min_max']*5+['z_scale']*2+['max_abs']*1),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_hybirdrnn_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def default_lstnet_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'optimizer': 'adam',
            'loss': Choice(['mae', 'mse', 'huber_loss']),
            'rnn_type': Choice(['gru', 'lstm']),
            'skip_rnn_type': Choice(['gru', 'lstm']),
            'cnn_filters': Choice([64]*2+[128]*3+[256]*2),
            'kernel_size': Choice([1]+[3]*3+[6]*3),
            'rnn_units': Choice([64]*2+[128]*3+[256]*2),
            'skip_rnn_units': Choice([32]*2+[64]*3+[128]*2),
            'rnn_layers': Choice([1]*1+[2]*4+[3]*4),
            'skip_rnn_layers': Choice([1]*1+[2]*4+[3]*4),
            'drop_rate': Choice([0., 0.1, 0.2]),
            'skip_period': Choice([0, 2, 3, 5]),
            'ar_order': Choice([2, 3, 5]),
            'forecast_length': Choice([1]*8+[3, 6]),
            'window': Choice(self.window if isinstance(self.window, list) else [self.window]),

            'y_log': Choice(['none-log']*4+['logx']*1),
            'y_scale': Choice(['min_max']*5+['z_scale']*2+['max_abs']*1),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_lstnet_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def default_nbeats_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'horizon': self.horizon,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'optimizer': 'adam',
            'nb_blocks_per_stack': Choice([1, 2, 3]),
            'hidden_layer_units': Choice([64, 128, 256]),
            'forecast_length': Choice([1]*8+[3, 6]),

            'y_log': Choice(['none-log']*4+['logx']*1),
            'y_scale': Choice(['min_max']*4+['z_scale']*1),
            'outlier': Choice(['none-outlier']*5+['clip']*3+['fill']*1)
        }

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_nbeats_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_deepar:
            univar_containers['deepar'] = (
                DeepARForecastEstimator, self.default_deepar_init_kwargs, self.default_deepar_fit_kwargs)
        if self.enable_hybirdrnn:
            univar_containers['hybirdrnn'] = (
                HybirdRNNGeneralEstimator, self.default_hybirdrnn_init_kwargs, self.default_hybirdrnn_fit_kwargs)
            multivar_containers['hybirdrnn'] = (
                HybirdRNNGeneralEstimator, self.default_hybirdrnn_init_kwargs, self.default_hybirdrnn_fit_kwargs)
        if self.enable_lstnet:
            univar_containers['lstnet'] = (
                LSTNetGeneralEstimator, self.default_lstnet_init_kwargs, self.default_lstnet_fit_kwargs)
            multivar_containers['lstnet'] = (
                LSTNetGeneralEstimator, self.default_lstnet_init_kwargs, self.default_lstnet_fit_kwargs)
        if self.enable_nbeats:
            univar_containers['nbeats'] = (
                NBeatsForecastEstimator, self.default_nbeats_init_kwargs, self.default_nbeats_fit_kwargs)
            multivar_containers['nbeats'] = (
                NBeatsForecastEstimator, self.default_nbeats_init_kwargs, self.default_nbeats_fit_kwargs)

        if self.task == consts.Task_UNIVARIATE_FORECAST:
            return univar_containers
        elif self.task == consts.Task_MULTIVARIATE_FORECAST:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIATE_FORECAST}'
                             f' or {consts.Task_MULTIVARIATE_FORECAST}.')


class DLClassRegressSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Deep Learning Search Space for Time Series Classification and Regression.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'univariate-binaryclass',
        'univariate-multiclass', 'multivariate-binaryclass, or ’multivariate-multiclass’.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support accuracy, f1, auc, recall, precision.
    enable_hybirdrnn: bool, default True.
    enable_lstnet: bool, default True.
    hybirdrnn_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which hybirdrnn is searched.
    lstnet_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which lstnet is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of deepar_init_kwargs, hybirdrnn_init_kwargs and lstnet_init_kwargs,
        you can refer to `hyperts.framework.estimators.HybirdRNNGeneralEstimator, and
        hyperts.framework.estimators.LSTNetGeneralEstimator.`
    2. If other parameters exist, set them directly. For example, n_estimators=200.
    """
    def __init__(self, task=None, timestamp=None, metrics=None,
                 enable_hybirdrnn=True,
                 enable_lstnet=True,
                 enable_inceptiontime=True,
                 hybirdrnn_init_kwargs=None,
                 lstnet_init_kwargs=None,
                 inceptiontime_init_kwargs=None,
                 **kwargs):
        if hasattr(kwargs, 'covariables'):
            kwargs.pop('covariables', None)
        if enable_hybirdrnn and hybirdrnn_init_kwargs is not None:
            kwargs['hybirdrnn_init_kwargs'] = hybirdrnn_init_kwargs
        if enable_lstnet and lstnet_init_kwargs is not None:
            kwargs['inceptiontime_init_kwargs'] = inceptiontime_init_kwargs
        if enable_inceptiontime and inceptiontime_init_kwargs is not None:
            kwargs['inceptiontime_init_kwargs'] = inceptiontime_init_kwargs
        super(DLClassRegressSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.metrics = metrics
        self.enable_hybirdrnn = enable_hybirdrnn
        self.enable_lstnet = enable_lstnet
        self.enable_inceptiontime = enable_inceptiontime

    @property
    def default_hybirdrnn_init_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'rnn_type': Choice(['gru', 'lstm']),
            'rnn_units': Choice([64]*2+[128]*3+[256]*2),
            'rnn_layers': Choice([1]*1+[2]*4+[3]*4+[4]*1),
            'drop_rate': Choice([0.]*4+[0.1]*4+[0.2]*1),

            'x_scale': Choice(['min_max']*8+['max_abs']*1+['z_scale']*1)
        }

    @property
    def default_hybirdrnn_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def default_lstnet_init_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'rnn_type': Choice(['gru', 'lstm']),
            'cnn_filters': Choice([64]*2+[128]*3+[256]*2),
            'kernel_size': Choice([1, 3, 5, 8]),
            'rnn_units': Choice([64]*2+[128]*3+[256]*2),
            'rnn_layers': Choice([1]*1+[2]*4+[3]*4+[4]*1),
            'drop_rate': Choice([0.]*4+[0.1]*4+[0.2]*1),
            'skip_period': 0,

            'x_scale': Choice(['min_max']*8+['max_abs']*1+['z_scale']*1)
        }

    @property
    def default_lstnet_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def default_inceptiontime_init_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            'blocks': Choice([1, 3, 6]),
            'cnn_filters': Choice([32, 64, 128, 256]),
            'short_filters': Choice([32, 64, 128]),

            'x_scale': Choice(['min_max']*8+['max_abs']*1+['z_scale']*1)
        }

    @property
    def default_inceptiontime_fit_kwargs(self):
        return {
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def estimators(self):
        class_containers = {}
        regress_containers = {}

        if self.enable_hybirdrnn:
            class_containers['hybirdrnn'] = (
                HybirdRNNGeneralEstimator, self.default_hybirdrnn_init_kwargs, self.default_hybirdrnn_fit_kwargs)
            regress_containers['hybirdrnn'] = (
                HybirdRNNGeneralEstimator, self.default_hybirdrnn_init_kwargs, self.default_hybirdrnn_fit_kwargs)
        if self.enable_lstnet:
            class_containers['lstnet'] = (
                LSTNetGeneralEstimator, self.default_lstnet_init_kwargs, self.default_lstnet_fit_kwargs)
            regress_containers['lstnet'] = (
                LSTNetGeneralEstimator, self.default_lstnet_init_kwargs, self.default_lstnet_fit_kwargs)
        if self.enable_inceptiontime:
            class_containers['inceptiontime'] = (
                InceptionTimeGeneralEstimator, self.default_inceptiontime_init_kwargs,
                self.default_inceptiontime_fit_kwargs)
            regress_containers['inceptiontime'] = (
                InceptionTimeGeneralEstimator, self.default_inceptiontime_init_kwargs,
                self.default_inceptiontime_fit_kwargs)

        if self.task in consts.TASK_LIST_CLASSIFICATION:
            return class_containers
        elif self.task in consts.TASK_LIST_REGRESSION:
            return regress_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.TASK_LIST_CLASSIFICATION}'
                             f', or {consts.TASK_LIST_REGRESSION}.')


class StatsDetectionSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Statistical Search Space for Time Series Anomaly Detection.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'detection'.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support f1, precision, recall, and so on.
    enable_iforest: bool, default True.
    enable_ocsvm: bool, default True.
    iforest_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which prophet is searched.
    ocsvm_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which prophet is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of iforest_init_kwargs, ocsvm_init_kwargs,
        you can refer to `hyperts.framework.estimators.IforestDetectionEstimator`,
        'hyperts.framework.estimators.OCSVMDetectionEstimator'.
    2. If other parameters exist, set them directly. For example, covariables=['is_holiday'].
    """
    def __init__(self, task=None, timestamp=None,
                 enable_iforest=True,
                 enable_ocsvm=True,
                 iforest_init_kwargs=None,
                 ocsvm_init_kwargs=None,
                 drop_observed_sample=False,
                 **kwargs):
        if enable_iforest and iforest_init_kwargs is not None:
            kwargs['iforest_init_kwargs'] = iforest_init_kwargs
        if enable_ocsvm and ocsvm_init_kwargs is not None:
            kwargs['ocsvm_init_kwargs'] = ocsvm_init_kwargs
        super(StatsDetectionSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.enable_iforest = enable_iforest
        self.enable_ocsvm = enable_ocsvm
        self.drop_observed_sample = drop_observed_sample

    @property
    def default_iforest_init_kwargs(self):
        default_init_kwargs = {
            'contamination': Choice([0.05] * 5 + [0.06, 0.07, 0.08, 0.09, 0.1]),
            'n_estimators': Choice([50, 100, 200, 500]),

            'x_scale': Choice(['min_max', 'z_scale'])
        }

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_iforest_fit_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'covariates': self.covariables,
        }

    @property
    def default_ocsvm_init_kwargs(self):
        default_init_kwargs = {
            # 'contamination': Choice([0.05] * 5 + [0.06, 0.07, 0.08, 0.09, 0.1]),
            'kernel': Choice(['linear', 'poly', 'sigmoid'] + ['rbf'] * 7),
            'tol': Choice([1e-5, 1e-3, 1e-2, 1e-1]),
            'nu': Choice([0.05, 0.1, 0.2, 0.5]),

            'x_scale': Choice(['min_max', 'z_scale'])
        }

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_ocsvm_fit_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'covariates': self.covariables,
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_iforest:
            univar_containers['iforest'] = (
            IForestDetectionEstimator, self.default_iforest_init_kwargs, self.default_iforest_fit_kwargs)
            multivar_containers['iforest'] = (
            IForestDetectionEstimator, self.default_iforest_init_kwargs, self.default_iforest_fit_kwargs)
        if self.enable_ocsvm:
            univar_containers['ocsvm'] = (
            OCSVMDetectionEstimator, self.default_ocsvm_init_kwargs, self.default_ocsvm_fit_kwargs)
            multivar_containers['ocsvm'] = (
            OCSVMDetectionEstimator, self.default_ocsvm_init_kwargs, self.default_ocsvm_fit_kwargs)

        if self.task == consts.Task_UNIVARIATE_DETECTION:
            return univar_containers
        elif self.task == consts.Task_MULTIVARIATE_DETECTION:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIATE_DETECTION}'
                             f' or {consts.Task_MULTIVARIATE_DETECTION}.')


class DLDetectionSearchSpace(BaseSearchSpaceGenerator, SearchSpaceMixin):
    """Deep Learning Search Space for Time Series Anomaly Detection.

    Parameters
    ----------
    task: str or None, optional, default None. If not None, it must be 'detection'.
    timestamp: str or None, optional, default None.
    metrics: str or None, optional, default None. Support f1, precision, recall, and so on.
    enable_convvae: bool, default True.
    convvae_init_kwargs: dict or None, optional, default None. If not None, you can customize
        the hyper-parameters by which prophet is searched.

    Returns
    ----------
    search space.

    Notes
    ----------
    1. For the hyper-parameters of iforest_init_kwargs, ocsvm_init_kwargs,
        you can refer to `hyperts.framework.estimators.ConvVAEDetectionEstimator`.
    2. If other parameters exist, set them directly. For example, covariables=['is_holiday'].
    """
    def __init__(self, task=None,
                 timestamp=None, metrics=None,
                 window=None, horizon=1,
                 enable_conv_vae=True,
                 conv_vae_init_kwargs=None,
                 drop_observed_sample=False,
                 **kwargs):
        if enable_conv_vae and conv_vae_init_kwargs is not None:
            kwargs['conv_vae_init_kwargs'] = conv_vae_init_kwargs
        super(DLDetectionSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.metrics = metrics
        self.window = window
        self.horizon = horizon
        self.enable_conv_vae = enable_conv_vae
        self.drop_observed_sample = drop_observed_sample

    @property
    def default_conv_vae_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,

            # 'contamination': Choice([0.05] * 5 + [0.06, 0.07, 0.08, 0.09, 0.1]),
            'latent_dim': Choice([4, 8, 12]),
            'conv_type': Choice(['general', 'separable']),
            'cnn_filters': Choice([16, 32, 64]),
            'nb_layers': Choice([2, 3]),
            'drop_rate': Choice([0.0] * 1 + [0.15] * 1 + [0.2] * 7 + [0.25] * 1),
            'window': Choice(self.window if isinstance(self.window, list) else [self.window]),

            'x_scale': Choice(['min_max'] * 1 + ['z_scale'] * 9)
        }

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if self.drop_observed_sample:
            default_init_kwargs['drop_sample_rate'] = Choice([0.0, 0.1, 0.2, 0.5, 0.8])

        return default_init_kwargs

    @property
    def default_conv_vae_fit_kwargs(self):
        return {
            'covariates': self.covariables,

            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_conv_vae:
            univar_containers['conv_vae'] = (
            ConvVAEDetectionEstimator, self.default_conv_vae_init_kwargs, self.default_conv_vae_fit_kwargs)
            multivar_containers['conv_vae'] = (
            ConvVAEDetectionEstimator, self.default_conv_vae_init_kwargs, self.default_conv_vae_fit_kwargs)

        if self.task == consts.Task_UNIVARIATE_DETECTION:
            return univar_containers
        elif self.task == consts.Task_MULTIVARIATE_DETECTION:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIATE_DETECTION}'
                             f' or {consts.Task_MULTIVARIATE_DETECTION}.')