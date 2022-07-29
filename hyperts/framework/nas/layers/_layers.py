import gc
import numpy as np
from hyperts.framework.dl import layers

from hypernets.core.search_space import ModuleSpace
from hypernets.core.ops import Choice, ModuleChoice, InputChoice


def compile_layer(search_space, layer_class, name, **kwargs):
    if kwargs.get('name') is None:
        kwargs['name'] = name

    cache = search_space.__dict__.get('weights_cache')
    if cache is not None:
        layer = cache.retrieve(kwargs['name'])
        if layer is None:
            layer = layer_class(**kwargs)
            cache.put(kwargs['name'], layer)
    else:
        layer = layer_class(**kwargs)

    return layer

class HyperLayer(ModuleSpace):
    def __init__(self, keras_layer_class, space=None, name=None, **hyperparams):
        self.keras_layer_class = keras_layer_class
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        self.keras_layer = compile_layer(self.space, self.keras_layer_class, self.name, **self.param_values)

    def _forward(self, inputs):
        return self.keras_layer(inputs)


class CalibrateSize(ModuleSpace):
    def __init__(self, node, name_prefix, space=None, name=None, **hyperparams):
        self.node = node
        self.name_prefix = name_prefix
        self.reduce0 = None
        self.reduce1 = None
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        self.compile_layer = compile_layer

    def factorized_reduce(self, name_posfix, period, filters, strides=1):
        return self.compile_layer(
            search_space=self.space,
            layer_class=layers.FactorizedReduce,
            period=period,
            filters=filters,
            strides=strides,
            name=f'{self.name_prefix}_factorized_reduce_{name_posfix}')

    def get_timestemp(self, x):
        return x.get_shape().as_list()[1]

    def get_channels(self, x):
        return x.get_shape().as_list()[-1]

    def _forward(self, inputs):
        if isinstance(inputs, list):
            t = [self.get_timestemp(inp) for inp in inputs]
            c = [self.get_channels(inp) for inp in inputs]
            min_t_value = int(np.min(t))
            min_c_value = int(np.min(c))

            x = inputs[self.node]
            if t[self.node] != min_t_value and self.reduce0 is None:
                self.reduce0 = self.factorized_reduce(f'timestemp{self.node}', min_t_value, min_c_value//2)
            if c[self.node] != min_c_value and self.reduce1 is None:
                self.reduce1 = self.factorized_reduce(f'variables{self.node}', min_t_value, min_c_value//2)
            if t[self.node] != min_t_value:
                x = self.reduce0(x)
            if c[self.node] != min_c_value:
                x = self.reduce1(x)

            return x
        else:
            return inputs

class SafeMerge(ModuleSpace):

    def __init__(self, name_prefix, ops='add', space=None, name=None, **hyperparams):
        self.ops = ops.lower()
        self.name_prefix = name_prefix
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        pass

    def _on_params_ready(self):
        pass

    def _forward(self, inputs):
        if isinstance(inputs, list):
            pv = self.param_values
            if pv.get('name') is None:
                pv['name'] = self.name
            if self.ops == 'add':
                return layers.Add(name=pv['name'])(inputs)
            elif self.ops == 'concat':
                return layers.Concatenate(**pv)(inputs)
            else:
                raise ValueError(f'Not supported operation:{self.ops}')
        else:
            return inputs

def stem_ops(input, units=64):
    rnn = HyperLayer(layers.GRU,
                     units=units,
                     return_sequences=True,
                     name='stem_gru')
    if input is None:
        input = rnn
    else:
        rnn(input)
    ln = HyperLayer(layers.LayerNormalization, name='stem_layernorm')(rnn)

    return ln, input

def cell_ops(inputs,
             name_prefix,
             block_no,
             node_no, cell_no,
             filters_or_units,
             kernel_size=(1, 3, 5)):
    name_prefix = f'{name_prefix}_block{block_no}_node{node_no}_cell{cell_no}'

    inpc = InputChoice(inputs, num_chosen_most=1, name=f'{name_prefix}_inputchoice')(inputs)

    if isinstance(filters_or_units, (tuple, list)):
        vaive_cnn_filters = Choice(list(filters_or_units))
        depsep_cnn_filters = Choice(list(filters_or_units))
        gru_filters = Choice(list(filters_or_units))
        lstm_filters = Choice(list(filters_or_units))
    else:
        vaive_cnn_filters = depsep_cnn_filters = gru_filters = lstm_filters = filters_or_units

    vaive_cnn = HyperLayer(layers.Conv1D,
                           filters=vaive_cnn_filters,
                           padding='same',
                           activation='relu',
                           kernel_size=Choice(list(kernel_size)),
                           name=f'{name_prefix}_conv1d')
    depsep_cnn = HyperLayer(layers.SeparableConv1D,
                            filters=depsep_cnn_filters,
                            padding='same',
                            activation='relu',
                            kernel_size=Choice(list(kernel_size)),
                            name=f'{name_prefix}_separableconv1d')
    gru = HyperLayer(layers.GRU,
                     units=gru_filters,
                     return_sequences=True,
                     name=f'{name_prefix}_gru')
    lstm = HyperLayer(layers.LSTM,
                      units=lstm_filters,
                      return_sequences=True,
                      name=f'{name_prefix}_lstm')
    identity = HyperLayer(layers.Identity, name=f'{name_prefix}_identity')

    op_choice = ModuleChoice([vaive_cnn, depsep_cnn, gru, lstm, identity], name=f'{name_prefix}_modulechoice')(inpc)

    return op_choice

def node_ops(inputs,
             name_prefix,
             block_no, node_no,
             filters_or_units=(16, 32, 64),
             kernel_size=(1, 3, 5)):
    cell0 = cell_ops(inputs, name_prefix, block_no, node_no, 0, filters_or_units, kernel_size)
    cell1 = cell_ops(inputs, name_prefix, block_no, node_no, 1, filters_or_units, kernel_size)

    out0 = CalibrateSize(node=0, name_prefix=f'{name_prefix}_block{block_no}_node{node_no}_reduce0')([cell0, cell1])
    out1 = CalibrateSize(node=1, name_prefix=f'{name_prefix}_block{block_no}_node{node_no}_reduce1')([cell0, cell1])

    out = merge_ops(inputs=[out0, out1], name_prefix=f'{name_prefix}_block{block_no}_node{node_no}')

    return out

def merge_ops(inputs, name_prefix, ops='add'):
    if ops == 'add':
        merge = HyperLayer(layers.Add, name=f'{name_prefix}_add')(inputs)
    elif ops == 'concat':
        merge = HyperLayer(layers.Concatenate, name=f'{name_prefix}_concat')(inputs)
    else:
        raise ValueError(f'Not supported operation:{ops}')
    return merge


class LayerWeightsCache(object):
    def __init__(self):
        self.reset()
        super(LayerWeightsCache, self).__init__()

    def reset(self):
        self.cache = dict()
        self.hit_counter = 0
        self.miss_counter = 0

    def clear(self):
        del self.cache
        gc.collect()
        self.reset()

    def hit(self):
        self.hit_counter += 1

    def miss(self):
        self.miss_counter += 1

    def put(self, key, layer):
        assert self.cache.get(key) is None, f'Duplicate keys are not allowed. key:{key}'
        self.cache[key] = layer

    def retrieve(self, key):
        item = self.cache.get(key)
        if item is None:
            self.miss()
        else:
            self.hit()
        return item