# -*- coding:utf-8 -*-

import numpy as np

from hypernets.tabular import column_selector
from hypernets.pipeline.base import DataFrameMapper
from hypernets.pipeline.base import Pipeline
from hypernets.pipeline.transformers import SimpleImputer
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace
from hypernets.core.ops import Choice, ModuleChoice


from hyperts.framework.wrappers import SimpleTSEstimator
from hyperts.framework.wrappers.stats_wrappers import ProphetWrapper, VARWrapper, TSForestWrapper
from hyperts.utils.transformers import TimeSeriesHyperTransformer

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.nas import layers as ops
from hyperts.framework.search_space import SearchSpaceMixin, HyperParams, WithinColumnSelector


##################################### Define NAS Search Space #####################################

class TSNASGenrealSearchSpace(SearchSpaceMixin):
    """TS-NAS Search Space for Time Series Forecasting, Classification and Regression.

    Parameters
    ----------
    num_blocks: int, the number of blocks.
    num_nodes: int, the number of nodes.
    init_filters_or_units: int, the number of filters(CNN) or units(RNN).
    block_ops: str, the operations for node of block, optional {'add', 'concat'}.

    Returns
    ----------
    search space.
    """
    def __init__(self, num_blocks=2, num_nodes=4, init_filters_or_units=64,
                 block_ops='concat', name='tsnas', drop_observed_sample=True, **kwargs):
        super(TSNASGenrealSearchSpace, self).__init__(**kwargs)
        self.name = name
        self.num_blocks = num_blocks
        self.num_nodes = num_nodes
        self.block_ops = block_ops
        self.init_filters_or_units = init_filters_or_units
        self.drop_observed_sample = drop_observed_sample

    @property
    def default_forecasting_init_kwargs(self):
        default_init_kwargs = {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'horizon': self.horizon,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,

            'forecast_length': Choice([1] * 8 + [3, 6]),
            'y_log': Choice(['none-log'] * 4 + ['logx'] * 1),
            'y_scale': Choice(['min_max'] * 4 + ['z_scale'] * 1),
            'outlier': Choice(['none-outlier'] * 5 + ['clip'] * 3 + ['fill'] * 1),
            'drop_sample_rate': Choice([0.0, 0.1, 0.2, 0.5, 0.8])}

        default_init_kwargs = self.initial_window_kwargs(default_init_kwargs)

        if not self.drop_observed_sample:
            default_init_kwargs.pop('drop_sample_rate')

        return default_init_kwargs

    @property
    def default_classification_regression_init_kwargs(self):
        return {
            'timestamp': self.timestamp,
            'task': self.task,
            'metrics': self.metrics,
            'reducelr_patience': 5,
            'earlystop_patience': 15,
            'summary': True,
            'epochs': consts.TRAINING_EPOCHS,
            'batch_size': None,
            'verbose': 1,

            'x_scale': Choice(['min_max']*8+['max_abs']*1+['z_scale']*1)
        }

    def __call__(self, *args, **kwargs):
        space = HyperSpace()
        with space.as_default():
            input = ops.HyperLayer(layers.Identity, name='relay_input')
            stem, input = ops.stem_ops(input, units=self.init_filters_or_units)
            ffa = ops.HyperLayer(layers.FeedForwardAttention, name='ffattention', return_sequences=True)
            identity = ops.HyperLayer(layers.Identity, name='choice_identity')
            head = ModuleChoice([ffa, identity], name='modulechoice_head')(stem)
            head = ops.HyperLayer(layers.Identity, name='head_identity')(head)

            node0 = stem
            node1 = head
            for block_no in range(self.num_blocks):
                inputs = [node0, node1]
                for node_no in range(self.num_nodes):
                    node = ops.node_ops(inputs,
                                        name_prefix=self.name,
                                        block_no=block_no,
                                        node_no=node_no)
                    inputs.append(node)

                if self.block_ops.lower() == 'concat':
                    outputs = inputs
                elif self.block_ops.lower() == 'add':
                    outputs = []
                    for i, input in enumerate(inputs):
                        outputs.append(ops.CalibrateSize(i, name_prefix=f'{self.name}_block{block_no}_reduce_out')(inputs))
                else:
                    raise ValueError(f'Not supported operation:{self.block_ops}')

                out = ops.SafeMerge(name_prefix=f'{self.name}_block{block_no}', ops=self.block_ops)(outputs)
                node0 = node1
                node1 = out

            out = ops.HyperLayer(layers.Activation, activation='relu', name=f'{self.name}_out_relu')(out)
            out = ops.HyperLayer(layers.GlobalAveragePooling1D, name=f'{self.name}_out_gap')(out)
            nas = ops.HyperLayer(layers.Dropout, rate=0.1, name=f'{self.name}_out_drouout')(out)

            if self.task in consts.TASK_LIST_FORECAST:
                default_nas_init_kwargs = self.default_forecasting_init_kwargs
            else:
                default_nas_init_kwargs = self.default_classification_regression_init_kwargs
            params = HyperParams(**default_nas_init_kwargs)(out)

        return space


#####################################  Define Simple Search Space #####################################

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