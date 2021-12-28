# -*- coding:utf-8 -*-

import time
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepMixin, BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


class HybirdRNNModel(Model, BaseDeepMixin):
    """
    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self,
                 task,
                 window,
                 rnn_type,
                 continuous_columns,
                 categorical_columns,
                 rnn_units,
                 rnn_layers,
                 drop_rate=0.,
                 nb_outputs=1,
                 nb_steps=1,
                 out_activation='linear',
                 summary=False,
                 **kwargs):

        super(HybirdRNNModel, self).__init__(**kwargs)
        self.task = task
        self.window = window
        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.drop_rate = drop_rate
        self.nb_outputs = nb_outputs
        self.nb_steps = nb_steps
        self.activation = out_activation

        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self._model = self._build(continuous_columns, categorical_columns)
        if summary:
            self._model.summary()

    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics
        return metrics

    def _build(self, continuous_columns, categorical_columns):
        K.clear_session()
        continuous_inputs, categorical_inputs = self.build_input_head(self.window, continuous_columns, categorical_columns)
        denses = self.build_denses(continuous_columns, continuous_inputs)
        embeddings = self.build_embeddings(categorical_columns, categorical_inputs)
        if embeddings is not None:
            x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
        else:
            x = denses

        # backbone
        x = self.rnn_forward(x, self.rnn_units, self.rnn_layers, self.rnn_type, name=self.rnn_type, drop_rate=self.drop_rate)
        outputs = self.build_output_tail(x, self.task, self.nb_outputs, self.nb_steps)

        if self.task in consts.TASK_LIST_FORECAST:
            outputs = layers.Activation(self.activation, name=f'output_activation_{self.activation}')(outputs)

        all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
        model = Model(inputs=all_inputs, outputs=[outputs], name=f'HybirdRNN-{self.rnn_type}')
        return model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self._model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.train_loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        results = {'loss': self.train_loss_tracker.result()}
        results.update({m.name: m.result() for m in self.metrics})
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self._model(x, training=False)
        loss = self.compiled_loss(y, y_pred)
        self.val_loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        results = {'loss': self.val_loss_tracker.result()}
        results.update({m.name: m.result() for m in self.metrics})
        return results


class HybirdRNN(BaseDeepEstimator):
    """

    """

    def __init__(self,
                 task,
                 rnn_type='gru',
                 rnn_units=16,
                 rnn_layers=1,
                 drop_rate=0.,
                 out_activation='linear',
                 timestamp=None,
                 window=7,
                 horizon=1,
                 forecast_length=1,
                 metrics='auto',
                 monitor='val_loss',
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
        self.monitor = monitor
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.reducelr_patience = reducelr_patience
        self.earlystop_patience = earlystop_patience
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(HybirdRNN, self).__init__(task,
                                        timestamp=timestamp,
                                        window=window,
                                        horizon=horizon,
                                        forecast_length=forecast_length,
                                        continuous_columns=continuous_columns,
                                        categorical_columns=categorical_columns)

    def _build_estimator(self, **kwargs):
        model = HybirdRNNModel(task=self.task,
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
        return model

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        if kwargs['epochs'] < 10:
            self.reducelr_patience = 0
            self.earlystop_patience = 0

        self._compile_info(self.monitor, self.reducelr_patience, self.earlystop_patience, self.learning_rate)

        self.model = self._build_estimator()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])

        if self.callbacks is not None:
            kwargs['callbacks'] = self.callbacks

        history = self.model.fit(x=train_X, y=train_y, validation_data=(valid_X, valid_y), **kwargs)

        return history

    def predict(self, X, batch_size=128):
        start = time.time()
        probs = self.predict_proba(X, batch_size)
        preds = self.proba2predict(probs, encode_to_label=True)
        logger.info(f'predict taken {time.time() - start}s')
        return preds

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model._model(X, training=False)