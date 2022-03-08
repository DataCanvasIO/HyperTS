# -*- coding:utf-8 -*-

import numpy as np

from hypernets.pipeline.base import DataFrameMapper
from hypernets.pipeline.base import Pipeline
from hypernets.pipeline.transformers import SimpleImputer
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.tabular import column_selector

from hyperts.framework.wrappers import SimpleTSEstimator
from hyperts.framework.wrappers.stats_wrappers import ProphetWrapper, VARWrapper, TSForestWrapper
from hyperts.utils.transformers import TimeSeriesHyperTransformer


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


def search_space_univariate_forecast_generator(covariate=(), timestamp=None):
    fit_kwargs = {'timestamp': timestamp}
    def search_space(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            dfm_inputs = []
            if len(covariate) > 0 or timestamp is not None:
                if len(covariate) > 0:
                    covariate_imputer = SimpleImputer(missing_values=np.nan,
                                                      strategy=impute_strategy,
                                                      name=f'covariate_imputer_{seq_no}',
                                                      force_output_as_float=True)
                    covariate_num_pipeline = Pipeline([covariate_imputer],
                                                  columns=WithinColumnSelector(column_selector.column_number_exclude_timedelta, covariate),
                                                  name=f'covariate_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(covariate_num_pipeline)
                if timestamp is not None:
                    time_series_pipeline = Pipeline([TimeSeriesHyperTransformer()],
                                                    columns=[timestamp],
                                                    name=f'default_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(time_series_pipeline)
                last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                                   df_out_dtype_transforms=[(column_selector.column_object, 'int')])(dfm_inputs)
            else:
                last_transformer = input
            SimpleTSEstimator(ProphetWrapper, fit_kwargs, interval_width=Choice([0.5, 0.6]), seasonality_mode=Choice(['additive', 'multiplicative']))(last_transformer)
            space.set_inputs(input)
        return space
    return search_space


def search_space_multivariate_forecast_generator(covariate=(), timestamp=None):
    fit_kwargs = {'timestamp': timestamp}
    def search_space(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            dfm_inputs = []
            if len(covariate) > 0 or timestamp is not None:
                if len(covariate) > 0:
                    covariate_imputer = SimpleImputer(missing_values=np.nan,
                                                      strategy=impute_strategy,
                                                      name=f'covariate_imputer_{seq_no}',
                                                      force_output_as_float=True)
                    covariate_pipeline = Pipeline([covariate_imputer],
                                                  columns=covariate,
                                                  name=f'covariate_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(covariate_pipeline)
                if timestamp is not None:
                    time_series_pipeline = Pipeline([TimeSeriesHyperTransformer()],
                                                    columns=[timestamp],
                                                    name=f'default_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(time_series_pipeline)
                last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                                   df_out_dtype_transforms=[(column_selector.column_object, 'int')])(dfm_inputs)
            else:
                last_transformer = input
            SimpleTSEstimator(VARWrapper, fit_kwargs, ic=Choice(['aic', 'fpe', 'hqic', 'bic']))(last_transformer)
            space.set_inputs(input)
        return space
    return search_space


def search_space_multivariate_classification():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        SimpleTSEstimator(TSForestWrapper, fit_kwargs=None, n_estimators=Choice([50, 100, 150]))(input)
        space.set_inputs(input)
    return space


# TODO:  define others search space

