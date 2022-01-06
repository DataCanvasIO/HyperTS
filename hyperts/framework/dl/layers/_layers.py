import math

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
            layers.Lambda(lambda k: k[:, -self.order:, :]),
            layers.Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1))),
            layers.Lambda(lambda k: K.reshape(k, (-1, self.order)))
        ])
        self.dense = layers.Dense(1, kernel_initializer='he_normal')

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
        self.nin = layers.Conv1D(nb_variables, 1, activation='relu')
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, inputs, **kwargs):
        x = self.nin(inputs)
        x = self.pool(x)
        return x

    def get_config(self):
        config = {'nb_variables': self.nb_variables}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def log_gaussian_loss(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(
        tf.math.log(tf.math.sqrt(2 * math.pi))
        + tf.math.log(sigma)
        + tf.math.truediv(tf.math.square(y_true - mu), 2 * tf.math.square(sigma)))


def build_input_head(window, continuous_columns, categorical_columns):
    """

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


def build_denses(continuous_columns, continuous_inputs, use_batchnormalization=False):
    """

    """
    if len(continuous_inputs) > 1:
        dense_layer = layers.Concatenate(name='concat_continuous_inputs')(
            list(continuous_inputs.values()))
    else:
        dense_layer = list(continuous_inputs.values())[0]

    if use_batchnormalization:
        dense_layer = layers.BatchNormalization(name='continuous_inputs_bn')(dense_layer)

    return dense_layer


def build_embeddings(categorical_columns, categorical_inputs):
    """

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
    """

    """
    if task in consts.TASK_LIST_REGRESSION + consts.TASK_LIST_BINARYCLASS:
        outputs = layers.Dense(units=1, activation='sigmoid', name='dense_out')(x)
    elif task in consts.TASK_LIST_MULTICLASS:
        outputs = layers.Dense(units=nb_outputs, activation='softmax', name='dense_out')(x)
    elif task in consts.TASK_LIST_FORECAST:
        outputs = layers.Dense(units=nb_outputs * nb_steps, activation='linear', name='dense_out')(x)
        outputs = layers.Lambda(lambda k: K.reshape(k, (-1, nb_steps, nb_outputs)), name='lambda_out')(outputs)
    else:
        raise ValueError(f'Unsupported task type {task}.')
    return outputs


def rnn_forward(x, nb_units, nb_layers, rnn_type, name, drop_rate=0., i=0):
    """

    """
    if rnn_type == 'lstm':
        for i in range(nb_layers - 1):
            x = layers.LSTM(units=nb_units, return_sequences=True, name=f'{name}_{i}')(x)
            x = layers.Dropout(rate=drop_rate, name=f'{name}_{i}_dropout')(x)
        x = layers.LSTM(units=nb_units, return_sequences=False, name=f'{name}_{i + 1}')(x)
    elif rnn_type == 'gru':
        for i in range(nb_layers - 1):
            x = layers.GRU(units=nb_units, return_sequences=True, name=f'{name}_{i}')(x)
            x = layers.Dropout(rate=drop_rate, name=f'{name}_{i}_dropout')(x)
        x = layers.GRU(units=nb_units, return_sequences=False, name=f'{name}_{i + 1}')(x)
    elif rnn_type == 'simple_rnn':
        for i in range(nb_layers - 1):
            x = layers.SimpleRNN(units=nb_units, return_sequences=True, name=f'{name}_{i}')(x)
            x = layers.Dropout(rate=drop_rate, name=f'{name}_{i}_dropout')(x)
        x = layers.SimpleRNN(units=nb_units, return_sequences=False, name=f'{name}_{i + 1}')(x)
    return x

def register_custom_objects(objs_dict:dict):
    for k,v in objs_dict.items():
        if custom_objects.get(k) is None:
            custom_objects[k] = v
        else:
            logger.error(f'`register_custom_objects` cannot register an existing key [{k}].')

custom_objects = {
    'MultiColEmbedding': MultiColEmbedding,
    'WeightedAttention': WeightedAttention,
    'AutoRegressive': AutoRegressive,
    'Highway': Highway,
    'log_gaussian_loss': log_gaussian_loss,
}