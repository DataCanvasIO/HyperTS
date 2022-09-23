# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl import BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return K.arange(0, horizon) / horizon

def NBeatsModel(task, continuous_columns, categorical_columns, window=10, nb_steps=1,
                stack_types=('trend', 'seasonality'), thetas_dim=(4, 8), nb_blocks_per_stack=3,
                share_weights_in_stack=False, hidden_layer_units=256, nb_outputs=1,
                nb_harmonics=None, out_activation='linear', **kwargs):
    """N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    continuous_columns: CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    window     : Positive Int - Length of the time series sequences for a sample,
                 i.e., backcast_length, default 10.
    nb_steps   : Positive Int -  The step length of forecast, i.e., forecast_length, default 1.
    stack_types : Tuple(Str) - Stack types, optional {'trend', 'seasonality', 'generic'}.
    thetas_dim  : Tuple(Int) - The number of units that make up each dense layer in each
                  block of every stack.
    nb_blocks_per_stack : Int - The number of block per stack.
    share_weights_in_stack : Bool - Whether to share weights in stack.
    hidden_layer_units : Int - The units of hidden layer.
    nb_outputs : Int, default 1.
    nb_harmonics : Int or None, -The number of harmonic terms for each stack type, default None.
    out_activation : Str - Forecast the task output activation function,
                 optional {'linear', 'sigmoid'}, default = 'linear'.

    References
    ----------
        [1] Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio, N-BEATS: Neural basis
            expansion analysis for interpretable time series forecasting, ICLR 2020.
        [2] https://github.com/philipperemy/n-beats.
    """
    if task not in consts.TASK_LIST_FORECAST:
        raise ValueError(f'Unsupported task type {task}.')

    weights = {}

    def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
        p = thetas.get_shape().as_list()[-1]
        if p % 2 == 0:
            p1, p2 = p // 2, p // 2
        else:
            p1, p2 = p // 2, p // 2 + 1
        t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
        s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
        s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
        if p == 1:
            s = s2
        else:
            s = K.concatenate([s1, s2], axis=0)
        s = K.cast(s, np.float32)
        return K.dot(thetas, s)

    def trend_model(thetas, backcast_length, forecast_length, is_forecast):
        p = thetas.get_shape().as_list()[-1]
        t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
        t = K.transpose(K.stack([t ** i for i in range(p)]))
        t = K.cast(t, np.float32)
        return K.dot(thetas, K.transpose(t))

    def create_block(x, e, stack_id, block_id, stack_type, nb_poly):
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        def reg(layer_with_weights):
            if share_weights_in_stack:
                layer_name = layer_with_weights.name.split('/')[-1]
                try:
                    reused_weights = weights[stack_id][layer_name]
                    return reused_weights
                except KeyError:
                    pass
                if stack_id not in weights:
                    weights[stack_id] = {}
                weights[stack_id][layer_name] = layer_with_weights
            return layer_with_weights

        backcast_, forecast_ = {}, {}

        fc1 = reg(layers.Dense(hidden_layer_units, activation='relu', name=n('fc1')))
        fc2 = reg(layers.Dense(hidden_layer_units, activation='relu', name=n('fc2')))
        fc3 = reg(layers.Dense(hidden_layer_units, activation='relu', name=n('fc3')))
        fc4 = reg(layers.Dense(hidden_layer_units, activation='relu', name=n('fc4')))

        if stack_type == 'generic':
            theta_b = reg(layers.Dense(nb_poly, use_bias=False, name=n('theta_b')))
            theta_f = reg(layers.Dense(nb_poly, use_bias=False, name=n('theta_f')))
            backcast = reg(layers.Dense(window, name=n('generic_backcast')))
            forecast = reg(layers.Dense(nb_steps, name=n('generic_forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(layers.Dense(nb_poly, use_bias=False, name=n('theta_f_b')))
            backcast = layers.Lambda(trend_model, arguments={'is_forecast': False, 'backcast_length': window,
                       'forecast_length': nb_steps}, name=n('trend_backcast'))
            forecast = layers.Lambda(trend_model, arguments={'is_forecast': True, 'backcast_length': window,
                       'forecast_length': nb_steps}, name=n('trend_forecast'))
        else: # 'seasonality'
            if nb_harmonics:
                theta_b = reg(layers.Dense(nb_harmonics, use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(layers.Dense(nb_steps, use_bias=False, name=n('theta_b')))
            theta_f = reg(layers.Dense(nb_steps, use_bias=False, name=n('theta_f')))
            backcast = layers.Lambda(seasonality_model, arguments={'is_forecast': False,
                       'backcast_length': window, 'forecast_length': nb_steps}, name=n('seasonality_backcast'))
            forecast = layers.Lambda(seasonality_model, arguments={'is_forecast': True,
                       'backcast_length': window, 'forecast_length': nb_steps}, name=n('seasonality_forecast'))
        for i in range(input_dim):
            if not bool(e) and exo_dim > 0:
                fc0 = layers.Concatenate(name=n(f'concat_x{i}_e'))([x[i]] + [e[j] for j in range(exo_dim)])
            else:
                fc0 = x[i]
            fc1_ = fc1(fc0)
            fc2_ = fc2(fc1_)
            fc3_ = fc3(fc2_)
            fc4_ = fc4(fc3_)
            theta_f_ = theta_f(fc4_)
            theta_b_ = theta_b(fc4_)
            backcast_[i] = backcast(theta_b_)
            forecast_[i] = forecast(theta_f_)

        return backcast_, forecast_

    K.clear_session()
    x_, y_, e_ = {}, {}, {}

    continuous_inputs, categorical_inputs = layers.build_input_head(window, continuous_columns, categorical_columns)
    x = layers.build_denses(continuous_columns, continuous_inputs)
    e = layers.build_embeddings(categorical_columns, categorical_inputs)

    input_dim = x.shape.as_list()[-1]

    if input_dim > nb_outputs:
        s = [nb_outputs, input_dim - nb_outputs]
        x, c = tf.split(x, num_or_size_splits=s, axis=-1, name='split_target_exo_inputs')
        input_dim = x.shape.as_list()[-1]
    else:
        c = None

    if e is not None:
        if c is not None:
            e = layers.Concatenate(axis=-1, name='concat_continuous_and_categorical_exo_inputs')([c, e])
        exo_dim = e.shape.as_list()[-1]
        for i in range(exo_dim):
            e_[i] = layers.Lambda(lambda k: k[..., i], name=f'lambda_decompose_exo_dim_{i}')(e)
    else:
        exo_dim = 0

    for i in range(input_dim):
        x_[i] = layers.Lambda(lambda k: k[..., i], name=f'lambda_decompose_input_dim_{i}')(x)

    for stack_id in range(len(stack_types)):
        stack_type = stack_types[stack_id]
        nb_poly = thetas_dim[stack_id]
        for block_id in range(nb_blocks_per_stack):
            backcast, forecast = create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
            for i in range(input_dim):
                x_[i] = layers.Subtract(name=f'x_sub_stack_{stack_id}_block_{block_id}_input_dim_{i}')([x_[i], backcast[i]])
                if stack_id == 0 and block_id == 0:
                    y_[i] = forecast[i]
                else:
                    y_[i] = layers.Add(name=f'y_sub_stack_{stack_id}_block_{block_id}_input_dim_{i}')([y_[i], forecast[i]])

    for i in range(input_dim):
        y_[i] = layers.Reshape((nb_steps, 1), name=f'y_reshape_input_dim_{i}')(y_[i])

    if nb_outputs > 1:
        outputs = layers.Concatenate(name='concat_y')([y_[i] for i in range(nb_outputs)])
    else:
        outputs = y_[0]

    if task in consts.TASK_LIST_FORECAST:
        outputs = layers.Activation(out_activation, name=f'output_activation_{out_activation}')(outputs)

    all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
    model = tf.keras.models.Model(inputs=all_inputs, outputs=[outputs], name=f'N-BEATS')

    return model


class NBeats(BaseDeepEstimator):
    """NBeats Estimator .

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    stack_types : Tuple(Str) - Stack types, optional {'trend', 'seasonality', 'generic'}.
                  default = ('trend', 'seasonality').
    thetas_dim  : Tuple(Int) - The number of units that make up each dense layer in each block of every stack.
                  default = (4, 8).
    nb_blocks_per_stack : Int - The number of block per stack.
                  default = 3.
    share_weights_in_stack : Bool - Whether to share weights in stack.
                  default = False.
    hidden_layer_units : Int - The units of hidden layer.
                  default = 256.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid', 'tanh'},
                 default = 'linear'.
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
                 stack_types=('trend', 'seasonality'),
                 thetas_dim=(4, 8),
                 nb_blocks_per_stack=3,
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 out_activation='linear',
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
        if task in consts.TASK_LIST_FORECAST and timestamp is None:
            raise ValueError('The forecast task requires [timestamp] name.')

        self.stack_types = stack_types
        self.thetas_dim = thetas_dim
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units
        self.out_activation = out_activation
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(NBeats, self).__init__(task=task,
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
            'stack_types': self.stack_types,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns,
            'thetas_dim': self.thetas_dim,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'share_weights_in_stack': self.share_weights_in_stack,
            'nb_outputs': self.meta.classes_,
            'nb_steps': self.forecast_length,
            'hidden_layer_units': self.hidden_layer_units,
            'out_activation': self.out_activation,
        }
        model_params = {**model_params, **self.model_kwargs, **kwargs}
        return NBeatsModel(**model_params)

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
            logger.info(f'Number of current NBeats params: {model.count_params()}')

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(train_ds, validation_data=valid_ds, **kwargs)

        return model, history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)