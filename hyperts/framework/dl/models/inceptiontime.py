# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl import BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def InceptionTimeModel(task, window, continuous_columns, categorical_columns, blocks=3,
            cnn_filters=32, bottleneck_size=32, kernel_size_list=(1, 3, 5, 8, 12),
            shortcut=True, short_filters=64, nb_outputs=1, **kwargs):
    """Inception Time Model (InceptionTime).

    Parameters
    ----------
    task       : Str - Only 'classification' is supported.
    window     : Positive Int - Length of the time series sequences for a sample.
    continuous_columns: CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    blocks      : Int - The depth of the net architecture.
    cnn_filters: Int - The number of cnn filters.
    bottleneck_size: Int - The number of bottleneck (a cnn layer).
    kernel_size_list: Tuple - The kernel size of cnn for a inceptionblock.
    shortcut   : Bool - Whether to use shortcut opration.
    short_filters: Int - The number of filters of shortcut conv1d layer.
    nb_outputs : Int - The number of classes, default 1.
    """
    if task not in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_REGRESSION:
        raise ValueError(f'Unsupported task type {task}.')

    kernel_size_list = list(filter(lambda x: x < window, kernel_size_list))

    K.clear_session()
    continuous_inputs, categorical_inputs = layers.build_input_head(window, continuous_columns, categorical_columns)
    denses = layers.build_denses(continuous_columns, continuous_inputs)
    embeddings = layers.build_embeddings(categorical_columns, categorical_inputs)
    if embeddings is not None:
        denses = layers.LayerNormalization(name='layer_norm')(denses)
        x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
    else:
        x = denses

    s = x
    for i in range(blocks):
        x = layers.InceptionBlock(filters=cnn_filters,
                                  bottleneck_size=bottleneck_size,
                                  kernel_size_list=kernel_size_list,
                                  name=f'inceptionblock_{i}')(x)
        if shortcut and i % 3 == 2:
            x = layers.Conv1D(short_filters, 1, padding='same', use_bias=False, name=f'shortconv_{i}')(x)
            s = layers.Shortcut(filters=short_filters, name=f'shortcut_{i}')(s)
            x = layers.Add(name=f'add_x_s_{i}')([x, s])
            x = layers.Activation('relu', name=f'relu_x_s_{i}')(x)
            s = x

    x = layers.GlobalAveragePooling1D(name='globeal_avg_pool')(x)
    outputs = layers.build_output_tail(x, task, nb_outputs)

    all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
    model = tf.keras.models.Model(inputs=all_inputs, outputs=[outputs], name=f'InceptionTime')

    return model


class InceptionTime(BaseDeepEstimator):
    """InceptionTime Estimator.

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    blocks     : Int - The depth of the net architecture.
                 default = 3.
    cnn_filters: Int - The number of cnn filters.
                 default = 32.
    bottleneck_size: Int - The number of bottleneck (a cnn layer).
                 default = 32.
    kernel_size_list: Tuple - The kernel size of cnn for a inceptionblock.
                 default = (1, 3, 5, 8, 12).
    shortcut   : Bool - Whether to use shortcut opration.
                 default = True.
    short_filters: Int - The number of filters of shortcut conv1d layer.
                 default = 64.
    timestamp  : Str or None - Timestamp name, the forecast task must be given,
                 default None.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 3.
    horizon    : Positive Int - Length of the prediction horizon,
                 default = 1.
    forecast_length : Positive Int - Step of the forecast outputs,
                 default = 1.
    metrics    : Str - List of metrics to be evaluated by the model during training and testing,
                 default = 'auto'.
    monitor_metric : Str - Quality indicators monitored during neural network training.
                 default = 'val_loss'.
    optimizer  : Str or keras Instance - for example, 'adam', 'sgd', and so on.
                 default = 'auto'.
    learning_rate : Positive Float - The optimizer's learning rate,
                 default = 0.001.
    loss       : Str - Loss function, for forecsting or regression, optional {'auto', 'mae', 'mse', 'huber_loss',
                 'mape'}, for classification, optional {'auto', 'categorical_crossentropy', 'binary_crossentropy},
                 default = 'auto'.
    reducelr_patience : Positive Int - The number of epochs with no improvement after which learning rate
                 will be reduced, default = 5.
    earlystop_patience : Positive Int - The number of epochs with no improvement after which training
                 will be stopped, default = 5.
    summary    : Bool - Whether to output network structure information,
                 default = True.
    continuous_columns: CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    """
    def __init__(self,
                 task,
                 blocks=3,
                 cnn_filters=32,
                 bottleneck_size=32,
                 kernel_size_list=(1, 3, 5, 8, 12),
                 shortcut=True,
                 short_filters=64,
                 timestamp=None,
                 window=3,
                 horizon=1,
                 forecast_length=1,
                 metrics='auto',
                 monitor_metric='val_loss',
                 optimizer='auto',
                 learning_rate=0.001,
                 loss='auto',
                 reducelr_patience=5,
                 earlystop_patience=10,
                 summary=True,
                 continuous_columns=None,
                 categorical_columns=None,
                 **kwargs):
        self.blocks = blocks
        self.cnn_filters = cnn_filters
        self.bottleneck_size = bottleneck_size
        self.kernel_size_list = kernel_size_list
        self.shortcut = shortcut
        self.short_filters = short_filters
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(InceptionTime, self).__init__(task=task,
                                            timestamp=timestamp,
                                            window=window,
                                            horizon=horizon,
                                            forecast_length=forecast_length,
                                            monitor_metric=monitor_metric,
                                            reducelr_patience=reducelr_patience,
                                            earlystop_patience=earlystop_patience,
                                            continuous_columns=continuous_columns,
                                            categorical_columns=categorical_columns)

    def _build_estimator(self, **kwargs):
        model_params = {
            'task': self.task,
            'window': self.window,
            'blocks': self.blocks,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns,
            'cnn_filters': self.cnn_filters,
            'bottleneck_size': self.bottleneck_size,
            'kernel_size_list': self.kernel_size_list,
            'nb_outputs': self.meta.classes_,
            'shortcut': self.shortcut,
            'short_filters': self.short_filters,
        }
        model_params = {**model_params, **self.model_kwargs, **kwargs}
        return InceptionTimeModel(**model_params)

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        train_ds = self._from_tensor_slices(X=train_X, y=train_y,
                                            batch_size=kwargs['batch_size'],
                                            epochs=kwargs['epochs'],
                                            shuffle=True)
        valid_ds = self._from_tensor_slices(X=valid_X, y=valid_y,
                                            batch_size=kwargs.pop('batch_size'),
                                            epochs=kwargs['epochs'],
                                            shuffle=False)
        model = self._build_estimator()

        if self.summary and kwargs['verbose'] != 0:
            model.summary()
        else:
            logger.info(f'Number of current InceptionTime params: {model.count_params()}')

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(train_ds, validation_data=valid_ds, **kwargs)

        return model, history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)