# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl import BaseDeepEstimator
from hyperts.framework.dl import BaseDeepDetectionMixin

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def vae_loss_funcion(input_x, decoder, z_mean, z_log_var, reconstruct_loss='mse', beta=0.1):
    """Loss = Reconstruction loss + Kullback-Leibler loss

    Parameters
    ----------
    input_x: Tensor, input time series.
        shape (batch_size, timestamp, features).
    decoder: Tensor, decoder reconstruct output.
        shape (batch_size, timestamp, features).
    z_mean: Tensor, latent representation mean of encoder.
        shape (batch_size, latent_dim).
    z_log_var: Tensor, latent representation log_var of encoder.
        shape (batch_size, latent_dim).
    reconstruct_loss: str, optional {'binary_crossentropy', ''mse'}.
    beta: positive float, a coefficient.
    """
    assert K.int_shape(input_x) == K.int_shape(decoder)
    if reconstruct_loss == 'binary_crossentropy':
        reconstruction_loss = K.sum(K.binary_crossentropy(input_x, decoder), axis=[1, 2])
    elif reconstruct_loss == 'mse':
        reconstruction_loss = K.sum(K.square(input_x - decoder), axis=[1, 2])
    else:
        raise ValueError(f'Unsupported {reconstruct_loss} loss function.')
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + beta * kl_loss)

    return vae_loss

def ConvVAEModel(task, window, nb_outputs, continuous_columns, categorical_columns,
             latent_dim=2, conv_type='general', cnn_filters=16, kernel_size=1, strides=1,
             nb_layers=2, activation='relu', drop_rate=0.0, out_activation='linear', **kwargs):
    """Variational AutoEncoder (VAE).

    Parameters
    ----------
    task          : Str - Only support anomaly detection.
                See hyperts.utils.consts for details.
    window        : Positive Int - Length of the time series sequences for a sample.
    nb_outputs    : Int, default 1.
    continuous_columns : CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns : CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    latent_dim    : Int - Latent representation of encoder, default 2.
    conv_type     : Str - Type of 1D convolution, optional {'general', 'separable'},
                default 'general'.
    cnn_filters   : Positive Int - The dimensionality of the output space (i.e. the number
        of filters in the convolution).
    kernel_size   : Positive Int - A single integer specifying the spatial dimensions
        of the filters,
    strides       : Int or tuple/list of a single integer - Specifying the stride length
        of the convolution.
    nb_layers     : Int - The layers of encoder and decoder, default 2.
    activation    : Str - The activation of hidden layers, default 'relu'.
    drop_rate     : Float between 0 and 1 - The rate of Dropout for neural nets.
    out_activation : Str - Forecast the task output activation function,
                 optional {'linear', 'sigmoid', 'tanh'}, default 'linear'.

    """
    if task not in consts.TASK_LIST_DETECTION:
        raise ValueError(f'Unsupported task type {task}.')

    if conv_type.lower() == 'general':
        ConvCell = layers.Conv1D
    elif conv_type.lower() == 'separable':
        ConvCell = layers.SeparableConv1D
    else:
        raise ValueError(f"Unsupported task type {conv_type}, optional {'general', 'separable'}")

    K.clear_session()

    continuous_inputs, categorical_inputs = layers.build_input_head(window, continuous_columns, categorical_columns)
    denses = layers.build_denses(continuous_columns, continuous_inputs)
    embeddings = layers.build_embeddings(categorical_columns, categorical_inputs)
    if embeddings is not None:
        denses = layers.LayerNormalization(name='layer_norm')(denses)
        x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
    else:
        x = denses

    if kernel_size > window:
        kernel_size = max(int(window // 3), 1)
        logger.warning(f'kernel_size cannot be larger than window, so it is reset to {kernel_size}.')

    conv_filters_list = []
    hidden_units = cnn_filters
    for i in range(nb_layers):
        conv_filters_list.append(cnn_filters)
        cnn_filters = cnn_filters // 2 if cnn_filters > 4 else cnn_filters

    # Encoder
    e = x
    for i, filters in enumerate(conv_filters_list):
        e = ConvCell(filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     activation=activation,
                     padding='same',
                     name=f'encoder_conv1d_{i}')(e)
        e = layers.Dropout(rate=drop_rate, name=f'encoder_dropout_rate{drop_rate}_{i}')(e)
    inter_shape = K.int_shape(e)
    e = layers.Flatten(name='encoder_flatten_conv')(e)
    e = layers.Dense(units=hidden_units,
                     activation=activation,
                     name='encoder_hidden_dense')(e)
    z_mean = layers.Dense(units=latent_dim, name='z_mean')(e)
    z_log_var = layers.Dense(units=latent_dim, name='z_log_var')(e)
    encoder = layers.Sampling(name='sampling')([z_mean, z_log_var])

    # Decoder
    d = layers.Dense(units=np.prod(inter_shape[1:]), name='decoder_inter')(encoder)
    d = layers.Reshape(target_shape=inter_shape[1:], name='decoder_reshape_inter')(d)
    for j, filters in enumerate(conv_filters_list[::-1]):
        d = layers.Conv1DTranspose(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   activation=activation,
                                   padding='same',
                                   name=f'decoder_conv1dtranspose_{j}')(d)
        d = layers.Dropout(rate=drop_rate, name=f'decoder_dropout_rate{drop_rate}_{j}')(d)
    d = layers.Conv1DTranspose(filters=nb_outputs,
                               kernel_size=kernel_size,
                               padding='same',
                               name='decoder_outputs')(d)
    decoder = layers.Activation(activation=out_activation,
                                name=f'decoder_activation_{out_activation}')(d)


    all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
    all_outputs = [decoder]
    model = tf.keras.models.Model(inputs=all_inputs, outputs=all_outputs, name='ConvVAE')

    # Loss
    if out_activation == 'sigmoid':
        reconstruct_loss = 'binary_crossentropy'
    else:
        reconstruct_loss = 'mse'
    vae_loss = vae_loss_funcion(input_x=x[:, :, :nb_outputs],
                                decoder=decoder,
                                z_mean=z_mean,
                                z_log_var=z_log_var,
                                reconstruct_loss=reconstruct_loss,
                                beta=0.1)

    model.add_loss(vae_loss)

    return model


class ConvVAE(BaseDeepEstimator, BaseDeepDetectionMixin):
    """Convolution Variational AutoEncoder Estimator (VAE).

    Parameters
    ----------
    task          : Str - Only support anomaly detection.
                See hyperts.utils.consts for details.
    timestamp     : Str or None - Timestamp name, the forecast task must be given,
                default None.
    window        : Positive Int - Length of the time series sequences for a sample.
    horizon       : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    latent_dim    : Int - Latent representation of encoder, default 2.
    conv_type     : Str - Type of 1D convolution, optional {'general', 'separable'},
                default 'general'.
    cnn_filters   : Positive Int - The dimensionality of the output space (i.e. the number
        of filters in the convolution).
    kernel_size   : Positive Int - A single integer specifying the spatial dimensions
        of the filters,
    strides       : Int or tuple/list of a single integer - Specifying the stride length
        of the convolution.
    nb_layers     : Int - The layers of encoder and decoder, default 2.
    activation    : Str - The activation of hidden layers, default 'relu'.
    drop_rate     : Float between 0 and 1 - The rate of Dropout for neural nets.
    out_activation : Str - Forecast the task output activation function,
                 optional {'linear', 'sigmoid', 'tanh'}, default 'linear'.
    metrics       : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor_metric : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer     : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary       : Bool - Whether to output network structure information,
                 default = True.
    continuous_columns : CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns : CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    """
    def __init__(self,
                 task,
                 timestamp,
                 contamination=0.05,
                 window=3,
                 horizon=1,
                 forecast_length=1,
                 latent_dim=2,
                 conv_type='separable',
                 cnn_filters=16,
                 kernel_size=1,
                 strides=1,
                 nb_layers=2,
                 activation='relu',
                 drop_rate=0.2,
                 out_activation='linear',
                 reconstract_dim=None,
                 metrics='auto',
                 monitor_metric='val_loss',
                 optimizer='auto',
                 learning_rate=0.001,
                 reducelr_patience=5,
                 earlystop_patience=10,
                 summary=True,
                 continuous_columns=None,
                 categorical_columns=None,
                 name='conv_vae',
                 **kwargs):
        if task not in consts.TASK_LIST_DETECTION:
            raise ValueError(f'Unsupported task type {task}.')

        self.latent_dim = latent_dim
        self.conv_type = conv_type
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.nb_layers = nb_layers
        self.activation = activation
        self.drop_rate =drop_rate
        self.out_activation = out_activation
        self.reconstract_dim = reconstract_dim
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(ConvVAE, self).__init__(task=task,
                                      timestamp=timestamp,
                                      window=window,
                                      horizon=horizon,
                                      forecast_length=forecast_length,
                                      monitor_metric=monitor_metric,
                                      reducelr_patience=reducelr_patience,
                                      earlystop_patience=earlystop_patience,
                                      continuous_columns=continuous_columns,
                                      categorical_columns=categorical_columns)
        self._update_mixin_params(name=name, contamination=contamination)

    def _bulid_estimator(self, **kwargs):
        if self.reconstract_dim is None:
            nb_outputs = self.meta.classes_
        else:
            nb_outputs = self.reconstract_dim
        model_params = {
            'task': self.task,
            'window': self.window,
            'nb_outputs': nb_outputs,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns,
            'latent_dim': self.latent_dim,
            'conv_type': self.conv_type,
            'cnn_filters': self.cnn_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'nb_layers': self.nb_layers,
            'activation': self.activation,
            'drop_rate': self.drop_rate,
            'out_activation': self.out_activation,
        }
        model_params = {**model_params, **self.model_kwargs, **kwargs}
        return ConvVAEModel(**model_params)

    def _fit(self, X_train, y_train, X_valid, y_valid, **kwargs):
        timestamp, reconstract_dim = y_train.shape[1], y_train.shape[2]

        train_ds = self._from_tensor_slices(X=X_train, y=y_train,
                                            batch_size=kwargs['batch_size'],
                                            epochs=kwargs['epochs'],
                                            shuffle=True)
        valid_ds = self._from_tensor_slices(X=X_valid, y=y_valid,
                                            batch_size=kwargs.pop('batch_size'),
                                            epochs=kwargs['epochs'],
                                            shuffle=False)
        model = self._bulid_estimator(**kwargs)

        if self.summary and kwargs['verbose'] != 0:
            model.summary()
        else:
            logger.info(f'Number of current ConvVAE params: {model.count_params()}')

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(train_ds, validation_data=valid_ds, **kwargs)

        y_pred = model.predict(X_train)

        self.decision_scores_ = self._pairwise_distances(
            y_true=y_train,
            y_pred=y_pred,
            n_samples=self.n_samples_,
            timestamp=timestamp,
            reconstract_dim=reconstract_dim)

        self._get_decision_attributes()

        return model, history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)
