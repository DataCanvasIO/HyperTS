# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def LSTNetModel(task, window, rnn_type, skip_rnn_type, continuous_columns, categorical_columns,
        cnn_filters, kernel_size, rnn_units, rnn_layers, skip_rnn_units, skip_rnn_layers, skip_period,
        ar_order, drop_rate=0., nb_outputs=1, nb_steps=1, out_activation='linear', summary=False, **kwargs):
    """Long-and Short-term Time-series Network Model (LSTNet).

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    window     : Positive Int - Length of the time series sequences for a sample.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}.
    skip_rnn_type : Str - Type of skip recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}, default = 'gru'.
    continuous_columns: CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    cnn_filters: Positive Int - The dimensionality of the output space (i.e. the number of filters
                 in the convolution).
    kernel_size: Positive Int - A single integer specifying the spatial dimensions of the filters,
    rnn_units  : Positive Int - The dimensionality of the output space for RNN.
    rnn_layers : Positive Int - The number of the layers for RNN.
    skip_rnn_units : Positive Int - The dimensionality of the output space for skip RNN.
    skip_rnn_layers : Positive Int - The number of the layers for skip RNN.
    skip_period: Positive Int or None - The length of skip for recurrent neural network.
    ar_order   : Positive Int or None - The window size of the autoregressive component.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets.
    nb_outputs : Int, default 1.
    nb_steps   : Int, The step length of forecast, default 1.
    out_activation : Str - Forecast the task output activation function,
                 optional {'linear', 'sigmoid'}, default = 'linear'.
    summary    : Bool - Whether to output network structure information,
                 default = True.

    """
    K.clear_session()
    continuous_inputs, categorical_inputs = layers.build_input_head(window, continuous_columns, categorical_columns)
    denses = layers.build_denses(continuous_columns, continuous_inputs)
    embeddings = layers.build_embeddings(categorical_columns, categorical_inputs)
    if embeddings is not None:
        x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
    else:
        x = denses

    # backbone
    if ar_order > window:
        ar_order = 1
        logger.warning('ar_order cannot be larger than window, so it is reset to 1.')

    if kernel_size > window:
        kernel_size = 1
        logger.warning('kernel_size cannot be larger than window, so it is reset to 1.')

    pt = int((window - kernel_size + 1) / skip_period) if skip_period > 0 else 0

    c = layers.SeparableConv1D(cnn_filters, kernel_size, activation='relu', name='conv1d')(x)
    c = layers.Dropout(rate=drop_rate, name='conv1d_dropout')(c)

    r = layers.rnn_forward(c, rnn_units, rnn_layers, rnn_type, name=rnn_type, drop_rate=drop_rate)
    r = layers.Lambda(lambda k: K.reshape(k, (-1, rnn_units)), name=f'lambda_{rnn_type}')(r)
    r = layers.Dropout(rate=drop_rate, name=f'lambda_{rnn_type}_dropout')(r)

    if skip_period*pt > 0:
        pt = max(int((window - kernel_size + 1) / skip_period), 1)
        s = layers.Lambda(lambda k: k[:, int(-pt*skip_period):, :], name=f'lambda_skip_{rnn_type}_0')(c)
        s = layers.Lambda(lambda k: K.reshape(k, (-1, pt, skip_period, cnn_filters)), name=f'lambda_skip_{rnn_type}_1')(s)
        s = layers.Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3)), name=f'lambda_skip_{rnn_type}_2')(s)
        s = layers.Lambda(lambda k: K.reshape(k, (-1, pt, cnn_filters)), name=f'lambda_skip_{rnn_type}_3')(s)

        s = layers.rnn_forward(s, skip_rnn_units, skip_rnn_layers, skip_rnn_type,
                                        name='skip_'+skip_rnn_type, drop_rate=drop_rate)
        s = layers.Lambda(lambda k: K.reshape(k, (-1, skip_period*skip_rnn_units)),
                                        name=f'lambda_skip_{rnn_type}_4')(s)
        s = layers.Dropout(rate=drop_rate, name=f'lambda_skip_{rnn_type}_dropout')(s)
        r = layers.Concatenate(name=f'{rnn_type}_concat_skip_{rnn_type}')([r, s])
    outputs = layers.build_output_tail(r, task, nb_outputs, nb_steps)

    if task in consts.TASK_LIST_FORECAST and nb_steps == 1 and ar_order > 0 and ar_order < window:
        z = layers.Lambda(lambda k: k[:, :, :nb_outputs], name='lambda_ar_0')(denses)
        z = layers.AutoRegressive(order=ar_order, nb_variables=nb_outputs, name='ar')(z)
        z = layers.Lambda(lambda k: K.reshape(k, (-1, 1, nb_outputs)), name='lambda_ar_1')(z)
        outputs = layers.Add(name='rnn_add_ar')([outputs, z])

    if task in consts.TASK_LIST_FORECAST:
        outputs = layers.Activation(out_activation, name=f'output_activation_{out_activation}')(outputs)

    all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
    model = Model(inputs=all_inputs, outputs=[outputs], name=f'LSTNet')
    if summary:
        model.summary()
    return model


class LSTNet(BaseDeepEstimator):
    """Long-and Short-term Time-series Network Estimator (LSTNet).

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}, default = 'gru'.
    skip_rnn_type : Str - Type of skip recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}, default = 'gru'.
    cnn_filters: Positive Int - The dimensionality of the output space (i.e. the number of filters
                 in the convolution), default = 16.
    kernel_size: Positive Int - A single integer specifying the spatial dimensions of the filters,
                 default = 1.
    rnn_units  : Positive Int - The dimensionality of the output space for recurrent neural network,
                 default = 16.
    rnn_layers : Positive Int - The number of the layers for recurrent neural network,
                 default = 1.
    skip_rnn_units : Positive Int - The dimensionality of the output space for skip recurrent neural network,
                 default = 16.
    skip_rnn_layers : Positive Int - The number of the layers for skip recurrent neural network,
                 default = 1.
    skip_period: Positive Int or None - The length of skip for recurrent neural network,
                 default = None.
    ar_order   : Positive Int or None - The window size of the autoregressive component,
                 default = None.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets,
                 default = 0.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid'},
                 default = 'linear'.
    timestamp  : Str or None - Timestamp name, the forecast task must be given,
                 default None.
    window     : Positive Int - Length of the time series sequences for a sample,
                 default = 7.
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

    """

    def __init__(self,
                 task,
                 rnn_type='gru',
                 skip_rnn_type='gru',
                 cnn_filters=16,
                 kernel_size=1,
                 rnn_units=16,
                 rnn_layers=1,
                 skip_rnn_units=16,
                 skip_rnn_layers=1,
                 skip_period=0,
                 ar_order=0,
                 drop_rate=0.,
                 out_activation='linear',
                 timestamp=None,
                 window=7,
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
        if task in consts.TASK_LIST_FORECAST and timestamp is None:
            raise ValueError('The forecast task requires [timestamp] name.')

        self.rnn_type = rnn_type
        self.skip_rnn_type = skip_rnn_type
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.skip_rnn_units = skip_rnn_units
        self.skip_rnn_layers = skip_rnn_layers
        self.skip_period = skip_period
        self.ar_order = ar_order
        self.drop_rate = drop_rate
        self.out_activation = out_activation
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(LSTNet, self).__init__(task=task,
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
        return LSTNetModel(task=self.task,
                           window=self.window,
                           rnn_type=self.rnn_type,
                           skip_rnn_type=self.skip_rnn_type,
                           continuous_columns=self.continuous_columns,
                           categorical_columns=self.categorical_columns,
                           cnn_filters=self.cnn_filters,
                           kernel_size=self.kernel_size,
                           rnn_units=self.rnn_units,
                           rnn_layers=self.rnn_layers,
                           skip_rnn_units=self.skip_rnn_units,
                           skip_rnn_layers=self.skip_rnn_layers,
                           skip_period=self.skip_period,
                           ar_order=self.ar_order,
                           drop_rate=self.drop_rate,
                           nb_outputs=self.mata.classes_,
                           nb_steps=self.forecast_length,
                           out_activation=self.out_activation,
                           summary=self.summary,
                           **kwargs)

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

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(train_ds, validation_data=valid_ds, **kwargs)

        return model, history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)