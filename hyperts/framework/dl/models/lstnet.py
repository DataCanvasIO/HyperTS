# -*- coding:utf-8 -*-

import time
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepMixin, BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


class LSTNetModel(Model, BaseDeepMixin):
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
                 skip_rnn_type,
                 continuous_columns,
                 categorical_columns,
                 cnn_filters,
                 kernel_size,
                 rnn_units,
                 rnn_layers,
                 skip_rnn_units,
                 skip_rnn_layers,
                 skip,
                 ar_order,
                 drop_rate=0.,
                 nb_outputs=1,
                 nb_steps=1,
                 out_activation='linear',
                 summary=False,
                 **kwargs):

        super(LSTNetModel, self).__init__(**kwargs)
        if window < kernel_size + 1:
            raise ValueError(f'Window(sequence length) must be greater than kernel_size.')

        self.task = task
        self.window = window
        self.rnn_type = rnn_type
        self.skip_rnn_type = skip_rnn_type
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.skip_rnn_units = skip_rnn_units
        self.skip_rnn_layers = skip_rnn_layers
        self.skip = skip
        self.ar_order = ar_order
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
        c = layers.SeparableConv1D(self.cnn_filters, self.kernel_size, activation='relu', name='conv1d')(x)
        c = layers.Dropout(rate=self.drop_rate, name='conv1d_dropout')(c)

        r = self.rnn_forward(c, self.rnn_units, self.rnn_layers, self.rnn_type, name=self.rnn_type, drop_rate=self.drop_rate)
        r = layers.Lambda(lambda k: K.reshape(k, (-1, self.rnn_units)), name=f'lambda_{self.rnn_type}')(r)
        r = layers.Dropout(rate=self.drop_rate, name=f'lambda_{self.rnn_type}_dropout')(r)

        if self.skip:
            pt = int((self.window - self.kernel_size + 1) / self.skip)
            s = layers.Lambda(lambda k: k[:, int(-pt*self.skip):, :], name=f'lambda_skip_{self.rnn_type}_0')(c)
            s = layers.Lambda(lambda k: K.reshape(k, (-1, pt, self.skip, self.cnn_filters)), name=f'lambda_skip_{self.rnn_type}_1')(s)
            s = layers.Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3)), name=f'lambda_skip_{self.rnn_type}_2')(s)
            s = layers.Lambda(lambda k: K.reshape(k, (-1, pt, self.cnn_filters)), name=f'lambda_skip_{self.rnn_type}_3')(s)

            s = self.rnn_forward(s, self.skip_rnn_units, self.skip_rnn_layers, self.skip_rnn_type,
                                            name='skip_'+self.rnn_type, drop_rate=self.drop_rate)
            s = layers.Lambda(lambda k: K.reshape(k, (-1, self.skip*self.skip_rnn_units)),
                                            name=f'lambda_skip_{self.rnn_type}_4')(s)
            s = layers.Dropout(rate=self.drop_rate, name=f'lambda_skip_{self.rnn_type}_dropout')(s)
            r = layers.Concatenate(name=f'{self.rnn_type}_concat_skip_{self.rnn_type}')([r, s])
        outputs = self.build_output_tail(r, self.task, self.nb_outputs, self.nb_steps)

        if self.task in consts.TASK_LIST_FORECAST and self.nb_steps == 1 and self.ar_order > 0:
            z = layers.Lambda(lambda k: k[:, :, :self.nb_outputs], name='lambda_ar_0')(denses)
            z = layers.AutoRegressive(order=self.ar_order, nb_variables=self.nb_outputs, name='ar')(z)
            z = layers.Lambda(lambda k: K.reshape(k, (-1, 1, self.nb_outputs)), name='lambda_ar_1')(z)
            outputs = layers.Add(name='rnn_add_ar')([outputs, z])

        if self.task in consts.TASK_LIST_FORECAST:
            outputs = layers.Activation(self.activation, name=f'output_activation_{self.activation}')(outputs)

        all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
        model = Model(inputs=all_inputs, outputs=[outputs], name=f'LSTNet')
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


class LSTNet(BaseDeepEstimator):
    """

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
                 skip=None,
                 ar_order=None,
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
        self.skip_rnn_type = skip_rnn_type
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.skip_rnn_units = skip_rnn_units
        self.skip_rnn_layers = skip_rnn_layers
        self.skip = skip
        self.ar_order = ar_order
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

        super(LSTNet, self).__init__(task=task,
                                     timestamp=timestamp,
                                     window=window,
                                     horizon=horizon,
                                     forecast_length=forecast_length,
                                     continuous_columns=continuous_columns,
                                     categorical_columns=categorical_columns)

    def _build_estimator(self, **kwargs):
        model = LSTNetModel(task=self.task,
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
                            skip=self.skip,
                            ar_order=self.ar_order,
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