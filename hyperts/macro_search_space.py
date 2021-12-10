# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from hyperts.config import Config as cfg
from hyperts.utils import consts
from hyperts.utils.transformers import TimeSeriesHyperTransformer
from hyperts.estimators import (ProphetForecastEstimator,
                                ARIMAForecastEstimator,
                                VARForecastEstimator,
                                TSFClassificationEstimator)

from hypernets.tabular import column_selector as tcs
from hypernets.core.ops import HyperInput, ModuleChoice, Optional
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.pipeline.transformers import (SimpleImputer,
                                             StandardScaler,
                                             MinMaxScaler,
                                             MaxAbsScaler,
                                             SafeOrdinalEncoder,
                                             AsTypeTransformer)

from hypernets.pipeline.base import Pipeline, DataFrameMapper
from hypernets.utils import logging, get_params

logger = logging.get_logger(__name__)


##################################### Define Data Proprecessing Pipeline #####################################
class WithinColumnSelector:

    def __init__(self, selector, selected_cols):
        self.selector = selector
        self.selected_cols = selected_cols

    def __call__(self, df):
        intersection = set(df.columns.tolist()).intersection(self.selected_cols)
        if len(intersection) > 0:
            selected_df = df[intersection]
            return self.selector(selected_df)
        else:
            return []


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
        raise NotImplementedError

    def create_preprocessor(self, hyper_input, options):
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)
        covariables = options.pop('covariables', None)
        timestamp = options.pop('timestamp', None)
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
            elif self.task in consts.TASK_LIST_FORECAST:
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

class StatsForecastSearchSpace(BaseSearchSpaceGenerator):

    def __init__(self, task, timestamp=None,
                 enable_prophet=True,
                 enable_arima=True,
                 enable_var=True,
                 **kwargs):
        kwargs['timestamp'] = timestamp
        logger.info("Tip: If other parameters exist, set them directly. For example, covariables=['is_holiday'].")

        super(StatsForecastSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.enable_prophet = enable_prophet
        self.enable_arima = enable_arima
        self.enable_var = enable_var

    @property
    def default_prophet_init_kwargs(self):
        return {
            # 'seasonality_prior_scale': Choice([True, False]),
            # 'yearly_seasonality': Choice([True, False]),
            # 'weekly_seasonality': Choice([True, False]),
            # 'daily_seasonality': Choice([True, False]),
            'seasonality_mode': Choice(['additive', 'multiplicative']),
            'n_changepoints': Choice([25, 35, 45]),
            'interval_width': Choice([0.6, 0.7, 0.8])
        }

    @property
    def default_prophet_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def default_arima_init_kwargs(self):
        return {
            'p': Choice([1, 2, 3, 4, 5]),
            'd': Choice([0, 1, 2]),
            'q': Choice([0, 1, 2, 3, 4, 5]),
            'trend': Choice(['n', 'c', 't', 'ct']),
            'seasonal_order': Choice([(0, 0, 0, 0), (1, 0, 1, 7), (1, 0, 2, 7),
                                      (2, 0, 1, 7), (2, 0, 2, 7), (1, 1, 1, 7), (0, 1, 1, 7)]),
            'y_scale': Choice(['min_max', 'max_abs', 'identity']),
            'y_log': Choice(['logx', 'identity'])
        }

    @property
    def default_arima_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def default_var_init_kwargs(self):
        return {
            'ic': Choice(['aic', 'fpe', 'hqic', 'bic']),
            'trend': Choice(['c', 'ct', 'ctt', 'nc', 'n']),
            'y_scale': Choice(['min_max', 'max_abs', 'identity']),
            'y_log': Choice(['logx', 'none'])
        }

    @property
    def default_var_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_prophet:
            univar_containers['prophet'] = (
            ProphetForecastEstimator, self.default_prophet_init_kwargs, self.default_prophet_fit_kwargs)
        if self.enable_arima:
            univar_containers['arima'] = (
            ARIMAForecastEstimator, self.default_arima_init_kwargs, self.default_arima_fit_kwargs)
        if self.enable_var:
            multivar_containers['var'] = (
            VARForecastEstimator, self.default_var_init_kwargs, self.default_var_fit_kwargs)

        if self.task == consts.Task_UNIVARIABLE_FORECAST:
            return univar_containers
        elif self.task == consts.Task_MULTIVARIABLE_FORECAST:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIABLE_FORECAST}'
                             f' or {consts.Task_MULTIVARIABLE_FORECAST}.')


class StatsClassificationSearchSpace(BaseSearchSpaceGenerator):

    def __init__(self, task, timestamp=None,
                 enable_tsf=True,
                 **kwargs):
        if hasattr(kwargs, 'covariables'):
            kwargs.pop('covariables', None)
        logger.info("Tip: If other parameters exist, set them directly. For example, n_estimators=200.")

        super(StatsClassificationSearchSpace, self).__init__(task, **kwargs)

        self.task = task
        self.timestamp = timestamp
        self.enable_tsf = enable_tsf

    @property
    def default_tsf_init_kwargs(self):  # Time Series Forest
        return {
            'min_interval': Choice([3, 5, 7]),
            'n_estimators': Choice([50, 100, 200, 300]),
        }

    @property
    def default_tsf_fit_kwargs(self):
        return {
            'timestamp': self.timestamp
        }

    @property
    def estimators(self):
        univar_containers = {}
        multivar_containers = {}

        if self.enable_tsf:
            univar_containers['tsf'] = (TSFClassificationEstimator, self.default_tsf_init_kwargs, self.default_tsf_fit_kwargs)

        if self.task in [consts.Task_UNIVARIABLE_BINARYCLASS, consts.Task_UNIVARIABLE_MULTICALSS]:
            return univar_containers
        elif self.task in [consts.Task_MULTIVARIABLE_BINARYCLASS, consts.Task_MULTIVARIABLE_MULTICALSS]:
            return multivar_containers
        else:
            raise ValueError(f'Incorrect task name, default {consts.Task_UNIVARIABLE_BINARYCLASS}'
                             f', {consts.Task_UNIVARIABLE_MULTICALSS}, {consts.Task_MULTIVARIABLE_BINARYCLASS}'
                             f', or {consts.Task_MULTIVARIABLE_MULTICALSS}.')


stats_forecast_search_space = StatsForecastSearchSpace

stats_classification_search_space = StatsClassificationSearchSpace

stats_regression_search_space = None

dl_univariate_forecast_search_space = None
dl_multivariate_forecast_search_space = None
dl_univariate_classification_search_space = None
dl_multivariate_classification_search_space = None


if __name__ == '__main__':
    from hypernets.searchers.random_searcher import RandomSearcher

    sfss = stats_forecast_search_space(task='univariable-forecast', timestamp='ts', covariables=['id', 'cos'])
    searcher = RandomSearcher(sfss, optimize_direction='min')
    sample = searcher.sample()
    print(sample)