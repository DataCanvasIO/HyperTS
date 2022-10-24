from hyperts.utils import consts
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class MultiColEmbedding(layers.Layer):
    """Multi Columns Embedding base on Embedding.

    Parameters
    ----------
    input_dims: Integer. A list or tuple. Sizes of the vocabulary for per columns.
    output_dims: A list or tuple. Dimension of the dense embedding.
    input_length: Integer. Length of input sequences.
    """

    def __init__(self, input_dims, output_dims, input_length=None, **kwargs):
        super(MultiColEmbedding, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_length = input_length
        self.embeddings = {}
        for i, (input_dim, output_dim) in enumerate(zip(self.input_dims, self.output_dims)):
            self.embeddings[i] = layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                                                  input_length=self.input_length)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.int32)
        embeddings = []
        for i in range(inputs.shape[-1]):
            sub_inp = inputs[:, :, i]
            embeddings.append(self.embeddings[i](sub_inp))
        outputs = tf.concat(embeddings, axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0], sum(self.output_dims))

    def get_config(self):
        config = {'input_dims': self.input_dims,
                  'output_dims': self.output_dims,
                  'input_length': self.input_length}
        base_config = super(MultiColEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedAttention(layers.Layer):
    """Weighted Attention based on softmax-Dense.

    Parameters
    ----------
    timesteps: time steps.
    input: (none, timesteps, nb_variables)
    output: (none, timesteps, nb_variables)
    """

    def __init__(self, timesteps, **kwargs):
        super(WeightedAttention, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.dense = layers.Dense(timesteps, activation='softmax')

    def call(self, inputs, **kwargs):
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.dense(x)
        x_weights = tf.transpose(x, perm=[0, 2, 1])
        x = tf.multiply(inputs, x_weights)
        return x

    def get_config(self):
        config = {'timesteps': self.timesteps}
        base_config = super(WeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForwardAttention(layers.Layer):
    """Feed Forward Attention.
       Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

    input: (none, timesteps, nb_variables)
    output: (none, nb_variables) or (none, timesteps, nb_variables)

    Parameters
    ----------
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    use_bias: Boolean. Whether the layer uses a bias vector. Default: `False`.
    activation: Str. Activation function to use. Default: `tanh`.

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(FFAttention())
    """

    def __init__(self, return_sequences=False, use_bias=True, activation='tanh', **kwargs):
        super(FeedForwardAttention, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.activation = activation
        self.return_sequences = return_sequences

        self.dense = layers.Dense(units=1, activation=activation, use_bias=use_bias)

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = tf.squeeze(x, axis=-1)
        x = tf.nn.softmax(x)
        x = tf.expand_dims(x, axis=-1)
        x = tf.multiply(inputs, x)
        if self.return_sequences:
            return x
        else:
            return tf.reduce_sum(x, axis=1)

    def get_config(self):
        config = {'use_bias': self.use_bias,
                  'return_sequences': self.return_sequences,
                  'activation': self.activation}
        base_config = super(FeedForwardAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AutoRegressive(layers.Layer):
    """AutoRegressive Layer based on Dense.

    Parameters
    ----------
    order: lookback step for original sequence.
    input: (none, timesteps, nb_variables)
    output: (none, nb_variables)
    """

    def __init__(self, order, nb_variables, **kwargs):
        super(AutoRegressive, self).__init__(**kwargs)
        self.order = order
        self.nb_variables = nb_variables
        self.transform = models.Sequential([
            layers.Lambda(self._cut_period),
            layers.Lambda(self._permute_dimensions),
            layers.Lambda(self._out_reshape)
        ])
        self.dense = layers.Dense(1)

    def _cut_period(self, x):
        return x[:, -self.order:, :]

    def _permute_dimensions(self, x):
        return K.permute_dimensions(x, (0, 2, 1))

    def _out_reshape(self, x):
        return K.reshape(x, (-1, self.order))

    def call(self, inputs, **kwargs):
        x = self.transform(inputs)
        x = self.dense(x)
        x = tf.reshape(x, (-1, self.nb_variables))
        return x

    def get_config(self):
        config = {'order': self.order, 'nb_variables': self.nb_variables}
        base_config = super(AutoRegressive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Highway(layers.Layer):
    """Highway to implement skip conenction based on NIN and GlobalAveragePooling1D.

    Parameters
    ----------
    input: (none, timesteps, nb_variables)
    output: (none, nb_variables)
    """

    def __init__(self, nb_variables, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.nb_variables = nb_variables
        self.nin = layers.Conv1D(nb_variables, 1, activation='relu', kernel_initializer='he_normal')
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, inputs, **kwargs):
        x = self.nin(inputs)
        x = self.pool(x)
        return x

    def get_config(self):
        config = {'nb_variables': self.nb_variables}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Time2Vec(layers.Layer):
    """The vector representation for time series.

    Parameters
    ----------
    kernel_size: int, dimension of vector embedding.
    periodic_activation: str, optional {'sin', 'cos'}, default 'sin'.
    input: (none, timesteps, nb_variables)
    output: (none, timesteps, kernel_size+nb_variables)
    """
    def __init__(self, kernel_size=32, periodic_activation='sin', **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        self.k = kernel_size
        self.actvition = periodic_activation

    def build(self, input_shape):
        # trend
        self.trend_weights = self.add_weight(name='trend_weights',
                                 shape=(1, input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.trend_bias = self.add_weight(name='trend_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        # periodic
        self.periodic_weights = self.add_weight(name='periodic_weights',
                                 shape=(input_shape[-1], self.k),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.periodic_bias = self.add_weight(name='periodic_bias',
                                 shape=(1, self.k),
                                 initializer='zeros',
                                 trainable=True)

        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        trend =  inputs * self.trend_weights + self.trend_bias
        if self.actvition.startswith('sin'):
            periodic = K.sin(K.dot(inputs, self.periodic_weights) + self.periodic_bias)
        elif self.actvition.startswith('cos'):
            periodic = K.cos(K.dot(inputs, self.periodic_weights) + self.periodic_bias)
        else:
            periodic1 = K.sin(K.dot(inputs, self.periodic_weights) + self.periodic_bias)
            periodic2 = K.cos(K.dot(inputs, self.periodic_weights) + self.periodic_bias)
            periodic = periodic1 + periodic2

        return K.concatenate([trend, periodic], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))

    def get_config(self):
        config = {'kernel_size': self.k}
        base_config = super(Time2Vec, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RevInstanceNormalization(layers.Layer):
    """Reversible Instance Normalization for Accurate Time-Series Forecasting
       against Distribution Shift, ICLR2022.

    Parameters
    ----------
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
    """
    def __init__(self, eps=1e-5, affine=True, **kwargs):
        super(RevInstanceNormalization, self).__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        self.affine_weight = self.add_weight(name='affine_weight',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=True)

        self.affine_bias = self.add_weight(name='affine_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(RevInstanceNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mode = kwargs.get('mode', None)
        if mode == 'norm':
            self._get_statistics(inputs)
            x = self._normalize(inputs)
        elif mode == 'denorm':
            x = self._denormalize(inputs)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, len(x.shape) - 1))
        self.mean = K.stop_gradient(K.mean(x, axis=dim2reduce, keepdims=True))
        self.stdev = K.stop_gradient(K.sqrt(K.var(x, axis=dim2reduce, keepdims=True) + self.eps))
        print(self.stdev)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def get_config(self):
        config = {'eps': self.eps,
                  'affine': self.affine}
        base_config = super(RevInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Identity(layers.Layer):
    """Identity Layer.

    """
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs


class Shortcut(layers.Layer):
    """ Shortcut Layer.

    Parampers
    ----------
    filters: the dimensionality of the output space for Conv1D.
    """
    def __init__(self, filters, activation='relu', **kwargs):
        super(Shortcut, self).__init__(**kwargs)
        self.filters = filters
        self.activation = activation
        self.conv = layers.Conv1D(filters, kernel_size=1, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x

    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(Shortcut, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InceptionBlock(layers.Layer):
    """InceptionBlock for time series.

    Parampers
    ----------
    filters: the dimensionality of the output space for Conv1D.
    kernel_size_list: list or tuple, a list of kernel size for Conv1D.
    strides: int or tuple, default 1.
    use_bottleneck: bool, whether to use bottleneck, default True.
    bottleneck_size: int, if use bottleneck, bottleneck_size is 32(default).
    activation: str, activation function, default 'relu'.
    """
    def __init__(self,
                 filters=32,
                 kernel_size_list=(1, 3, 5, 8, 12),
                 strides=1,
                 use_bottleneck=True,
                 bottleneck_size=32,
                 activation='linear',
                 **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size_list = kernel_size_list
        self.strides = strides
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.activation = activation

        if use_bottleneck:
            self.head = layers.Conv1D(bottleneck_size, 1, padding='same', activation=activation, use_bias=False)
        else:
            self.head = Identity()

        self.conv_list = []
        for kernel_size in kernel_size_list:
            self.conv_list.append(layers.Conv1D(filters=filters,
                                         kernel_size=kernel_size,
                                         padding='same',
                                         activation=activation,
                                         use_bias=False))

        self.max_pool = layers.MaxPool1D(pool_size=3, strides=1, padding='same')
        self.pool_conv = layers.Conv1D(filters, kernel_size=1, padding='same', activation=activation, use_bias=False)

        self.concat = layers.Concatenate(axis=2)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation(activation='relu')

    def call(self, inputs, **kwargs):
        x = self.head(inputs)
        convs = [conv(x) for conv in self.conv_list]
        pool = self.max_pool(x)
        pool_conv = self.pool_conv(pool)
        convs.append(pool_conv)
        x = self.concat(convs)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size_list': self.kernel_size_list,
                  'strides': self.strides,
                  'use_bottleneck': self.use_bottleneck,
                  'bottleneck_size': self.bottleneck_size,
                  'activation': self.activation}
        base_config = super(InceptionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FactorizedReduce(layers.Layer):
    """Factorized reduce for timestemp or variable.

    Parampers
    ----------
    period: lookback step for original sequence.
    filters: the dimensionality of the output space for Conv1D.
    strides: int or tuple, default 1.
    """
    def __init__(self, period, filters, strides=1, **kwargs):
        super(FactorizedReduce, self).__init__(**kwargs)
        self.period = period
        self.filters = filters
        self.strides = strides
        self.cropping1 = models.Sequential([
            layers.Conv1D(filters, kernel_size=1, strides=strides, use_bias=False),
            layers.Lambda(self._cut_period)
        ])
        self.cropping2 = models.Sequential([
            layers.ZeroPadding1D(padding=(0, 1)),
            layers.Cropping1D(cropping=(1, 0)),
            layers.Conv1D(filters, kernel_size=1, strides=strides, use_bias=False),
            layers.Lambda(self._cut_period)
        ])
        self.concat = layers.Concatenate(axis=-1)
        self.bn = layers.BatchNormalization()

    def _cut_period(self, x):
        return x[:, -self.period:, :]

    def call(self, inputs, **kwargs):
        c1 = self.cropping1(inputs)
        c2 = self.cropping2(inputs)
        x = self.concat([c1, c2])
        x = self.bn(x)

        return x

    def get_config(self):
        config = {'period': self.period,
                  'filters': self.filters,
                  'strides': self.strides}
        base_config = super(FactorizedReduce, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(layers.Layer):
    """Reparametrisation by sampling from Gaussian  N(0,I).

    Parampers
    ----------
    
    """
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = {}
        base_config = super(Sampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_input_head(window, continuous_columns, categorical_columns):
    """Build the input head. An input variable may have two parts: continuous variables
       and categorical variables.

    Parameters
    ----------
    window: length of the time series sequences for a sample, i.e., timestamp.
    continuous_columns: CategoricalColumn class.
        Contains some information(name, column_names, input_dim, dtype,
        input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
        Contains some information(name, vocabulary_size, embedding_dim,
        dtype, input_name) about categorical variables.
    """

    continuous_inputs = OrderedDict()
    categorical_inputs = OrderedDict()
    for column in continuous_columns:
        continuous_inputs[column.name] = layers.Input(shape=(window, column.input_dim),
                                                      name=column.name, dtype=column.dtype)

    if categorical_columns is not None and len(categorical_columns) > 0:
        categorical_inputs['all_categorical_vars'] = layers.Input(shape=((window, len(categorical_columns))),
                                                                  name='input_categorical_vars_all')

    return continuous_inputs, categorical_inputs


def build_denses(continuous_columns, continuous_inputs, use_layernormalization=False):
    """Concatenate continuous inputs.

    Parameters
    ----------
    continuous_columns: CategoricalColumn class.
        Contains some information(name, column_names, input_dim, dtype,
        input_name) about continuous variables.
    continuous_inputs: list, tf.keras.layers.Input objects.
    use_layernormalization: bool, default False.
    """

    if len(continuous_inputs) > 1:
        dense_layer = layers.Concatenate(name='concat_continuous_inputs')(
            list(continuous_inputs.values()))
    else:
        dense_layer = list(continuous_inputs.values())[0]

    if use_layernormalization:
        dense_layer = layers.LayerNormalization(name='continuous_inputs_ln')(dense_layer)

    return dense_layer


def build_embeddings(categorical_columns, categorical_inputs):
    """Build embeddings if there are categorical variables.

    Parameters
    ----------
    categorical_columns: CategoricalColumn class.
        Contains some information(name, vocabulary_size, embedding_dim,
        dtype, input_name) about categorical variables.
    categorical_inputs: list, tf.keras.layers.Input objects.
    """

    if 'all_categorical_vars' in categorical_inputs:
        input_layer = categorical_inputs['all_categorical_vars']
        input_dims = [column.vocabulary_size for column in categorical_columns]
        output_dims = [column.embedding_dim for column in categorical_columns]
        embeddings = MultiColEmbedding(input_dims, output_dims)(input_layer)
    else:
        embeddings = None

    return embeddings


def build_output_tail(x, task, nb_outputs, nb_steps=1):
    """Build the output tail.

    Parameters
    ----------
    task: str, See hyperts.utils.consts for details.
    nb_outputs: int, the number of output units.
    nb_steps: int, the step length of forecast, default 1.
    """

    if task in consts.TASK_LIST_REGRESSION + consts.TASK_LIST_BINARYCLASS:
        outputs = layers.Dense(units=1, activation='sigmoid', name='dense_out')(x)
    elif task in consts.TASK_LIST_MULTICLASS:
        outputs = layers.Dense(units=nb_outputs, activation='softmax', name='dense_out')(x)
    elif task in consts.TASK_LIST_FORECAST:
        outputs = layers.Dense(units=nb_outputs*nb_steps, activation='linear', name='dense_out')(x)
        outputs = layers.Lambda(lambda k: K.reshape(k, (-1, nb_steps, nb_outputs)), name='lambda_out')(outputs)
    else:
        raise ValueError(f'Unsupported task type {task}.')
    return outputs


def rnn_forward(x, nb_units, nb_layers, rnn_type, name, drop_rate=0., i=0, activation='tanh', return_sequences=False):
    """Multi-RNN layers.

    Parameters
    ----------
    nb_units: int, the dimensionality of the output space for
        recurrent neural network.
    nb_layers: int, the number of the layers for recurrent neural network.
    rnn_type: str, type of recurrent neural network,
        including {'basic', 'gru', 'lstm'}.
    name: recurrent neural network name.
    drop_rate: float, the rate of Dropout for neural nets, default 0.
    return_sequences: bool, whether to return the last output. in the output
        sequence, or the full sequence. default False.
    """

    RnnCell = {'lstm': layers.LSTM, 'gru': layers.GRU, 'basic': layers.SimpleRNN}[rnn_type]
    for i in range(nb_layers - 1):
        x = RnnCell(units=nb_units,
                    activation=activation,
                    return_sequences=True,
                    name=f'{name}_{i}')(x)
        if drop_rate > 0. and drop_rate < 0.5:
            x = layers.Dropout(rate=drop_rate, name=f'{name}_{i}_dropout')(x)
        elif drop_rate >= 0.5:
            x = layers.BatchNormalization(name=f'{name}_{i}_norm')(x)
    x = RnnCell(units=nb_units,
                activation=activation,
                return_sequences=return_sequences,
                name=f'{name}_{i+1}')(x)
    return x


layers_custom_objects = {
    'MultiColEmbedding': MultiColEmbedding,
    'WeightedAttention': WeightedAttention,
    'FeedForwardAttention': FeedForwardAttention,
    'AutoRegressive': AutoRegressive,
    'Highway': Highway,
    'Time2Vec': Time2Vec,
    'RevInstanceNormalization': RevInstanceNormalization,
    'Identity': Identity,
    'Shortcut': Shortcut,
    'InceptionBlock': InceptionBlock,
    'FactorizedReduce': FactorizedReduce,
    'Sampling': Sampling,
}