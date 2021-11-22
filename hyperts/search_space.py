# -*- coding:utf-8 -*-

import numpy as np

from hypergbm.pipeline import DataFrameMapper
from hypergbm.pipeline import Pipeline
from hypergbm.sklearn.transformers import SimpleImputer
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.tabular.column_selector import column_object
from hyperts.estimators import TSEstimatorMS, ProphetWrapper, VARWrapper, SKTimeWrapper


def search_space_univariate_forecast_generator(covariate=()):
    def f(dataframe_mapper_default=False, impute_strategy='mean', seq_no=0):
        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            if len(covariate) > 0:
                covariate_pipeline = Pipeline([SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                                   name=f'covariate_imputer_{seq_no}', force_output_as_float=True)],
                                    columns=covariate, name=f'covariate_pipeline_simple_{seq_no}')(input)
                last_transformer = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                                   df_out_dtype_transforms=[(column_object, 'int')])([covariate_pipeline])
            else:
                last_transformer = input
            TSEstimatorMS(ProphetWrapper, interval_width=Choice([0.5, 0.6]), seasonality_mode=Choice(['additive', 'multiplicative']))(last_transformer)
            space.set_inputs(input)
        return space
    return f


def search_space_multivariate_forecast():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(VARWrapper, ic=Choice(['aic', 'fpe', 'hqic', 'bic']))(input)
        space.set_inputs(input)
    return space


def space_classification_classification():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(SKTimeWrapper, n_estimators=Choice([50, 100, 150]))(input)
        space.set_inputs(input)
    return space


# TODO:  define others search space

