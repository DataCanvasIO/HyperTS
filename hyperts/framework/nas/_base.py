import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl import BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


class TSNASEstimator(BaseDeepEstimator):
    """TS-NAS Estimator.

    Parameters
    ----------
    task       : Str - Support forecast, classification, and regression.
                 See hyperts.utils.consts for details.
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

        self.out_activation = out_activation
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        self.space_sample = None

        super(TSNASEstimator, self).__init__(task=task,
                                        timestamp=timestamp,
                                        window=window,
                                        horizon=horizon,
                                        forecast_length=forecast_length,
                                        monitor_metric=monitor_metric,
                                        reducelr_patience=reducelr_patience,
                                        earlystop_patience=earlystop_patience,
                                        continuous_columns=continuous_columns,
                                        categorical_columns=categorical_columns)

    def build_inputs(self):
        K.clear_session()
        continuous_inputs, categorical_inputs = layers.build_input_head(self.window,
                                                                        self.continuous_columns,
                                                                        self.categorical_columns)
        denses = layers.build_denses(self.continuous_columns, continuous_inputs)
        embeddings = layers.build_embeddings(self.categorical_columns, categorical_inputs)
        if embeddings is not None:
            denses = layers.LayerNormalization(name='layer_norm')(denses)
            x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
        else:
            x = denses
        return continuous_inputs, categorical_inputs, x

    def _build_estimator(self, **kwargs):
        continuous_inputs, categorical_inputs, x = self.build_inputs()

        space, outputs = self.space_sample.compile_and_forward(x)
        self.space_sample = None

        outputs = layers.build_output_tail(outputs[0], self.task, self.meta.classes_, self.forecast_length)

        if self.task in consts.TASK_LIST_FORECAST:
            outputs = layers.Activation(self.out_activation, name=f'output_activation_{self.out_activation}')(outputs)

        all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
        model = tf.keras.models.Model(inputs=all_inputs, outputs=[outputs], name=f'TS-NAS')

        return model

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
            logger.info(f'Number of current TS-NAS params: {model.count_params()}')

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(train_ds, validation_data=valid_ds, **kwargs)

        return model, history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)
