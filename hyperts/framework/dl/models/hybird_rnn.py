# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepEstimator


def HybirdRNNModel(task, window, rnn_type, continuous_columns, categorical_columns,
        rnn_units, rnn_layers, drop_rate=0., nb_outputs=1, nb_steps=1, out_activation='linear',
        summary=False, **kwargs):
    """SimpleRNN|GRU|LSTM Model (HybirdRNN).

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    window     : Positive Int - Length of the time series sequences for a sample.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}.
    continuous_columns: CategoricalColumn class.
                 Contains some information(name, column_names, input_dim, dtype,
                 input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
                 Contains some information(name, vocabulary_size, embedding_dim,
                 dtype, input_name) about categorical variables.
    rnn_units  : Positive Int - The dimensionality of the output space for RNN.
    rnn_layers : Positive Int - The number of the layers for RNN.
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
    x = layers.rnn_forward(x, rnn_units, rnn_layers, rnn_type, name=rnn_type, drop_rate=drop_rate)
    outputs = layers.build_output_tail(x, task, nb_outputs, nb_steps)

    if task in consts.TASK_LIST_FORECAST:
        outputs = layers.Activation(out_activation, name=f'output_activation_{out_activation}')(outputs)

    all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
    model = Model(inputs=all_inputs, outputs=[outputs], name=f'HybirdRNN-{rnn_type}')
    if summary:
        model.summary()
    return model


class HybirdRNN(BaseDeepEstimator):
    """SimpleRNN|GRU|LSTM Estimator (HybirdRNN).

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
    rnn_type   : Str - Type of recurrent neural network,
                 optional {'simple_rnn', 'gru', 'lstm}, default = 'gru'.
    rnn_units  : Positive Int - The dimensionality of the output space for recurrent neural network,
                 default = 16.
    rnn_layers : Positive Int - The number of the layers for recurrent neural network,
                 default = 1.
    drop_rate  : Float between 0 and 1 - The rate of Dropout for neural nets,
                 default = 0.
    out_activation : Str - Forecast the task output activation function, optional {'linear', 'sigmoid'},
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
                 rnn_type='gru',
                 rnn_units=16,
                 rnn_layers=1,
                 drop_rate=0.,
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

        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.drop_rate = drop_rate
        self.out_activation = out_activation
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(HybirdRNN, self).__init__(task=task,
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
        return HybirdRNNModel(task=self.task,
                              window=self.window,
                              rnn_type=self.rnn_type,
                              continuous_columns=self.continuous_columns,
                              categorical_columns=self.categorical_columns,
                              rnn_units=self.rnn_units,
                              rnn_layers=self.rnn_layers,
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