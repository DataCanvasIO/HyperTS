# -*- coding:utf-8 -*-

import numpy as np

from hypernets.pipeline.base import DataFrameMapper, HyperTransformer
from hypernets.pipeline.base import Pipeline
from hypernets.pipeline.transformers import SimpleImputer
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.tabular.column_selector import column_object
from hyperts.estimators import TSEstimatorMS, ProphetWrapper, VARWrapper, SKTimeWrapper
from hyperts.transformers import TimeSeriesHyperTransformer


def search_space_univariate_forecast_generator(covariate=(), time_series=None):
    def f(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            dfm_inputs = []
            if len(covariate) > 0 or time_series is not None:
                if len(covariate) > 0:
                    covariate_imputer = SimpleImputer(missing_values=np.nan,
                                                      strategy=impute_strategy,
                                                      name=f'covariate_imputer_{seq_no}',
                                                      force_output_as_float=True)
                    covariate_pipeline = Pipeline([covariate_imputer],
                                                  columns=covariate,
                                                  name=f'covariate_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(covariate_pipeline)
                if time_series is not None:
                    time_series_pipeline = Pipeline([TimeSeriesHyperTransformer()],
                                                    columns=[time_series],
                                                    name=f'default_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(time_series_pipeline)
                last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                                   df_out_dtype_transforms=[(column_object, 'int')])(dfm_inputs)
            else:
                last_transformer = input
            TSEstimatorMS(ProphetWrapper, interval_width=Choice([0.5, 0.6]), seasonality_mode=Choice(['additive', 'multiplicative']))(last_transformer)
            space.set_inputs(input)
        return space
    return f


def search_space_multivariate_forecast_generator(covariate=(), time_series=None):
    def f(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            dfm_inputs = []
            if len(covariate) > 0 or time_series is not None:
                if len(covariate) > 0:
                    covariate_imputer = SimpleImputer(missing_values=np.nan,
                                                      strategy=impute_strategy,
                                                      name=f'covariate_imputer_{seq_no}',
                                                      force_output_as_float=True)
                    covariate_pipeline = Pipeline([covariate_imputer],
                                                  columns=covariate,
                                                  name=f'covariate_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(covariate_pipeline)
                if time_series is not None:
                    time_series_pipeline = Pipeline([TimeSeriesHyperTransformer()],
                                                    columns=[time_series],
                                                    name=f'default_pipeline_simple_{seq_no}')(input)
                    dfm_inputs.append(time_series_pipeline)
                last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                                   df_out_dtype_transforms=[(column_object, 'int')])(dfm_inputs)
            else:
                last_transformer = input
            TSEstimatorMS(VARWrapper, ic=Choice(['aic', 'fpe', 'hqic', 'bic']))(last_transformer)
            space.set_inputs(input)
        return space
    return f


# def search_space_multivariate_forecast():
#     space = HyperSpace()
#     with space.as_default():
#         input = HyperInput(name='input1')
#
#         space.set_inputs(input)
#     return space


# def space_classification_classification_generator(covariate=(), time_series=None):
#     def f(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
#         space = HyperSpace()
#         with space.as_default():
#             input = HyperInput(name='input1')
#             dfm_inputs = []
#             if len(covariate) > 0 or time_series is not None:
#                 if len(covariate) > 0:
#                     covariate_imputer = SimpleImputer(missing_values=np.nan,
#                                                       strategy=impute_strategy,
#                                                       name=f'covariate_imputer_{seq_no}',
#                                                       force_output_as_float=True)
#                     covariate_pipeline = Pipeline([covariate_imputer],
#                                                   columns=covariate,
#                                                   name=f'covariate_pipeline_simple_{seq_no}')(input)
#                     dfm_inputs.append(covariate_pipeline)
#                 if time_series is not None:
#                     time_series_pipeline = Pipeline([TimeSeriesHyperTransformer()],
#                                                     columns=[time_series],
#                                                     name=f'default_pipeline_simple_{seq_no}')(input)
#                     dfm_inputs.append(time_series_pipeline)
#                 last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
#                                                    df_out_dtype_transforms=[(column_object, 'int')])(dfm_inputs)
#             else:
#                 last_transformer = input
#             TSEstimatorMS(SKTimeWrapper, n_estimators=Choice([50, 100, 150]))(last_transformer)
#             space.set_inputs(input)
#         return space
#     return f


def space_classification_classification():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(SKTimeWrapper, n_estimators=Choice([50, 100, 150]))(input)
        space.set_inputs(input)
    return space


# TODO:  define others search space

