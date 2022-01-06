# -*- coding:utf-8 -*-

import time
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl.models import Model, BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def HybirdRNNModel(task, window, rnn_type, continuous_columns, categorical_columns,
        rnn_units, rnn_layers, drop_rate=0., nb_outputs=1, nb_steps=1, out_activation='linear',
        summary=False, **kwargs):
    """
    Parameters
    ----------


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

        model = self._build_estimator()

        model = self._compile_model(model, self.optimizer, self.learning_rate)

        history = model.fit(x=train_X, y=train_y, validation_data=(valid_X, valid_y), **kwargs)
        return model, history

    def predict(self, X, batch_size=128):
        start = time.time()
        probs = self.predict_proba(X, batch_size)
        preds = self.proba2predict(probs, encode_to_label=True)
        logger.info(f'predict taken {time.time() - start}s')
        return preds

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)