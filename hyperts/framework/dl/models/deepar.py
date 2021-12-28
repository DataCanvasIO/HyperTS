# -*- coding:utf-8 -*-

import math
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepMixin, BaseDeepEstimator


class DeepARModel(Model, BaseDeepMixin):
    """

    Parameters
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
                 summary=False,
                 **kwargs):
        super(DeepARModel, self).__init__(**kwargs)
        if task not in consts.TASK_LIST_FORECAST:
            raise ValueError(f'Unsupported task type {task}.')
        if nb_outputs != 1:
            raise ValueError('DeepAR only support univariable forecast.')

        self.window = window
        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.drop_rate = drop_rate
        self.nb_outputs = nb_outputs
        self.nb_steps = nb_steps

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
        mu = layers.Dense(self.nb_outputs, activation='linear', name='dense_mu')(x)
        sigma = layers.Dense(self.nb_outputs, activation='softplus', name='dense_sigma')(x)

        all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
        model = Model(inputs=all_inputs, outputs=[mu, sigma], name='DeepAR')
        return model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            mu, sigma = self._model(x, training=True)
            loss = self.log_gaussian_loss(y, mu, sigma)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.train_loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, mu)
        results = {'loss': self.train_loss_tracker.result()}
        results.update({m.name: m.result() for m in self.metrics})
        return results

    def test_step(self, data):
        x, y = data
        mu, sigma = self._model(x, training=False)
        loss = self.log_gaussian_loss(y, mu, sigma)
        self.val_loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, mu)
        results = {'loss': self.val_loss_tracker.result()}
        results.update({m.name: m.result() for m in self.metrics})
        return results

    def log_gaussian_loss(self, y_true, mu, sigma):
        return tf.reduce_mean(
            tf.math.log(tf.math.sqrt(2 * math.pi))
            + tf.math.log(sigma)
            + tf.math.truediv(tf.math.square(y_true - mu), 2 * tf.math.square(sigma)))


class DeepAR(BaseDeepEstimator):
    """

    """

    def __init__(self,
                 task,
                 timestamp,
                 rnn_type='gru',
                 rnn_units=16,
                 rnn_layers=1,
                 drop_rate=0.,
                 window=7,
                 horizon=1,
                 forecast_length=1,
                 metrics='auto',
                 monitor='val_loss',
                 optimizer='auto',
                 learning_rate=0.001,
                 loss='log_gaussian_loss',
                 reducelr_patience=5,
                 earlystop_patience=10,
                 summary=True,
                 continuous_columns=None,
                 categorical_columns=None,
                 **kwargs):
        if task not in consts.TASK_LIST_FORECAST:
            raise ValueError(f'Unsupported task type {task}.')

        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.drop_rate = drop_rate
        self.metrics = metrics
        self.monitor = monitor
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.reducelr_patience = reducelr_patience
        self.earlystop_patience = earlystop_patience
        self.summary = summary
        self.model_kwargs = kwargs.copy()

        super(DeepAR, self).__init__(task=task,
                                     timestamp=timestamp,
                                     window=window,
                                     horizon=horizon,
                                     forecast_length=forecast_length,
                                     continuous_columns=continuous_columns,
                                     categorical_columns=categorical_columns)

    def _bulid_estimator(self, **kwargs):
        model = DeepARModel(task=self.task,
                            window=self.window,
                            rnn_type=self.rnn_type,
                            continuous_columns=self.continuous_columns,
                            categorical_columns=self.categorical_columns,
                            rnn_units=self.rnn_units,
                            rnn_layers=self.rnn_layers,
                            drop_rate=self.drop_rate,
                            nb_outputs=self.mata.classes_,
                            nb_steps=self.forecast_length,
                            summary=self.summary,
                            **kwargs)
        return model

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        if kwargs['epochs'] < 10:
            self.reducelr_patience = 0
            self.earlystop_patience = 0

        self._compile_info(self.monitor, self.reducelr_patience, self.earlystop_patience, self.learning_rate)

        self.model = self._bulid_estimator()
        self.model.compile(optimizer=self.optimizer, metrics=[self.metrics])

        if self.callbacks is not None:
            kwargs['callbacks'] = self.callbacks

        history = self.model.fit(x=train_X, y=train_y, validation_data=(valid_X, valid_y), **kwargs)
        return history

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        mu, sigma = self.model._model(X, training=False)
        return tf.expand_dims(mu, axis=1)