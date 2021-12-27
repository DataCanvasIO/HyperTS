import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K


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
            layers.Lambda(lambda k: k[:, -order:, :]),
            layers.Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1))),
            layers.Lambda(lambda k: K.reshape(k, (-1, order)))
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