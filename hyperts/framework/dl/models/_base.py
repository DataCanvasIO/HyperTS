# -*- coding:utf-8 -*-

import time
import math
import numpy as np
import pandas as pd
from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from hyperts.utils import consts, toolbox as tstb
from hyperts.framework.dl import layers
from hyperts.framework.dl.timeseries import from_array_to_timeseries
from hyperts.framework.dl.metainfo import MetaTSFprocessor, MetaTSCprocessor

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class BaseDeepMixin:

    def build_input_head(self, continuous_columns, categorical_columns):
        """

        """
        continuous_inputs = OrderedDict()
        categorical_inputs = OrderedDict()
        for column in continuous_columns:
            continuous_inputs[column.name] = layers.Input(shape=(None, column.input_dim),
                                                          name=column.name, dtype=column.dtype)

        if categorical_columns is not None and len(categorical_columns) > 0:
            categorical_inputs['all_categorical_vars'] = layers.Input(shape=((None, len(categorical_columns))),
                                                                      name='input_categorical_vars_all')

        return continuous_inputs, categorical_inputs

    def build_denses(self, continuous_columns, continuous_inputs, use_batchnormalization=False):
        """

        """
        if len(continuous_inputs) > 1:
            dense_layer = layers.Concatenate(name='concat_continuous_inputs')(
                list(continuous_inputs.values()))
        else:
            dense_layer = list(continuous_inputs.values())[0]

        if use_batchnormalization:
            dense_layer = layers.BatchNormalization(name='continuous_inputs_bn')(dense_layer)

        return dense_layer

    def build_embeddings(self, categorical_columns, categorical_inputs):
        """

        """
        if 'all_categorical_vars' in categorical_inputs:
            input_layer = categorical_inputs['all_categorical_vars']
            input_dims = [column.vocabulary_size for column in categorical_columns]
            output_dims = [column.embedding_dim for column in categorical_columns]
            embeddings = layers.MultiColEmbedding(input_dims, output_dims)(input_layer)
        else:
            embeddings = None

        return embeddings

    def build_output_tail(self, x):
        """

        """
        if self.task in consts.TASK_LIST_REGRESSION + consts.TASK_LIST_BINARYCLASS:
            outputs = layers.Dense(units=1, activation='sigmoid', name='dense_out')(x)
        elif self.task in consts.TASK_LIST_MULTICLASS:
            outputs = layers.Dense(units=self.nb_outputs, activation='softmax', name='dense_out')(x)
        elif self.task in consts.TASK_LIST_FORECAST:
            outputs = layers.Dense(units=self.nb_outputs * self.nb_steps, activation='linear', name='dense_out')(x)
            outputs = layers.Lambda(lambda k: K.reshape(k, (-1, self.nb_steps, self.nb_outputs)), name='lambda_out')(
                outputs)
        return outputs

    def rnn_forward(self, x, nb_units, nb_layers, i=0):
        """

        """
        if self.rnn_type == 'lstm':
            for i in range(nb_layers - 1):
                x = layers.LSTM(units=nb_units, return_sequences=True, name=f'lstm_{i}')(x)
            x = layers.LSTM(units=nb_units, return_sequences=False, name=f'lstm_{i + 1}')(x)
        elif self.rnn_type == 'gru':
            for i in range(nb_layers - 1):
                x = layers.GRU(units=nb_units, return_sequences=True, name=f'gru_{i}')(x)
            x = layers.GRU(units=nb_units, return_sequences=False, name=f'gru_{i + 1}')(x)
        elif self.rnn_type == 'simple_rnn':
            for i in range(nb_layers - 1):
                x = layers.SimpleRNN(units=nb_units, return_sequences=True, name=f'rnn_{i}')(x)
            x = layers.SimpleRNN(units=nb_units, return_sequences=False, name=f'rnn_{i + 1}')(x)
        return x


class BaseDeepEstimator(object):
    """

    """

    def __init__(self, task,
                 timestamp=None,
                 window=None,
                 horizon=None,
                 forecast_length=1,
                 embedding_output_dim=4,
                 continuous_columns=None,
                 categorical_columns=None):
        self.task = task
        self.timestamp = timestamp
        self.window = window
        self.horizon = horizon
        self.forecast_length = forecast_length
        self.embedding_output_dim=embedding_output_dim
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.time_columns = None
        self.forecast_start = None
        self.model = None
        self.callbacks = None

    def _build_estimator(self, **kwargs):
        raise NotImplementedError

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit(self,
            X,
            y,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.2,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        start = time.time()
        X, y = self._preprocessor(X, y)
        if validation_data is not None:
            validation_data = self.mata.transform(*validation_data)

        if validation_data is None:
            if self.task in consts.TASK_LIST_FORECAST:
                X, X_val, y, y_val = tstb.temporal_train_test_split(X, y, test_size=validation_split)
            else:
                X, X_val, y, y_val = tstb.random_train_test_split(X, y, test_size=validation_split)
        else:
            if len(validation_data) != 2:
                raise ValueError(f'Unexpected validation_data length, expected 2 but {len(validation_data)}.')
            X_val, y_val = validation_data[0], validation_data[1]

        if batch_size is None:
            batch_size = min(int(len(X) / 16), 128)

        if steps_per_epoch is None:
            steps_per_epoch = len(X) // batch_size
            if steps_per_epoch == 0:
                steps_per_epoch = 1
        if validation_steps is None:
            validation_steps = len(X_val) // batch_size - 1
            if validation_steps <= 1:
                validation_steps = 1

        X_train, y_train = self._dataloader(self.task, X, y, self.window, self.horizon, self.forecast_length,
                                            is_train=True)
        X_valid, y_valid = self._dataloader(self.task, X_val, y_val, self.window, self.horizon, self.forecast_length,
                                            is_train=False)

        history = self._fit(X_train, y_train, X_valid, y_valid, epochs=epochs, batch_size=batch_size,
                            initial_epoch=initial_epoch,
                            verbose=verbose, callbacks=callbacks, shuffle=shuffle, class_weight=class_weight,
                            sample_weight=sample_weight,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                            validation_batch_size=validation_batch_size,
                            validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
                            use_multiprocessing=use_multiprocessing)

        logger.info(f'Training finished, total taken {time.time() - start}s.')

        return history

    def forecast(self, X):
        """Infer Function
        Task: forecastion.
        """
        start = time.time()
        if self.timestamp in X.columns:
            steps = X.shape[0]
            X = X.drop([self.timestamp], axis=1)
        else:
            raise ValueError('X is missing the timestamp columns.')

        if steps < self.forecast_length:
            raise ValueError(f'Forecast steps {steps} cannot be'
                             f'less than forecast length {self.forecast_length}.')

        if X.shape[1] >= 1:
            X = self.mata.transform_X(X)
            X_cont_cols, X_cat_cols = [], []
            for c in X.columns:
                if c in self.mata.cont_column_names:
                    X_cont_cols.append(c)
                elif c in self.mata.cat_column_names:
                    X_cat_cols.append(c)
                else:
                    raise ValueError('Unknown column.')
            X = X[X_cont_cols + X_cat_cols].values

        futures = []
        data = self.forecast_start
        if X.shape[1] >= 1:
            continuous_length = len(self.mata.cont_column_names)
            categorical_length = len(self.mata.cat_column_names)
            for i in range(math.ceil(steps / self.forecast_length)):
                pred = self._predict(data)
                futures.append(pred.numpy())
                covariable = np.expand_dims(X[i:i + self.forecast_length], 0)
                forcast = np.concatenate([pred.numpy(), covariable], axis=-1)
                if categorical_length > 0:
                    data[0] = np.append(data[0], forcast[:, :, :continuous_length]).reshape(1, -1, continuous_length)
                    data[1] = np.append(data[1], forcast[:, :, continuous_length:]).reshape(1, -1, categorical_length)
                    data = [data[0][:, -self.window:, :], data[1][:, -self.window:, :]]
                else:
                    data = np.append(data, forcast).reshape(1, -1, continuous_length)
        else:
            for i in range(math.ceil(steps / self.forecast_length)):
                pred = self._predict(data)
                futures.append(pred.numpy())
                forcast = pred.numpy()
                data = np.append(data, forcast).reshape(1, -1, self.mata.labels_)
                data = data[:, -self.window:, :]

        logger.info(f'forecast taken {time.time() - start}s')
        return np.array(futures).reshape(steps, -1)

    def predict_proba(self, X, **kwargs):
        start = time.time()
        X = self.mata.transform_X(X)
        probs = self.model.predict(X, **kwargs)
        if probs.shape[-1] == 1:
            probs = np.hstack([1 - probs, probs])
        logger.info(f'predict_proba taken {time.time() - start}s')
        return probs

    def proba2predict(self, proba, encode_to_label=True):
        if self.task in consts.TASK_LIST_REGRESSION:
            return proba
        if proba is None:
            raise ValueError('[proba] can not be none.')
        if len(proba.shape) == 1:
            proba = proba.reshape((-1, 1))

        if proba.shape[-1] > 1:
            predict = proba.argmax(axis=-1)
        else:
            predict = (proba > 0.5).astype('int32')
        if encode_to_label:
            logger.info('reverse indicators to labels.')
            predict = self.mata.inverse_transform_y(predict)
        return predict

    def _compile_info(self, monitor='val_loss', reducelr_patience=0, earlystop_patience=0):
        if self.task in consts.TASK_LIST_MULTICLASS:
            if self.loss == None:
                self.loss = tf.keras.losses.CategoricalCrossentropy()
            if self.metrics == None:
                self.metrics = tf.keras.metrics.Accuracy()
        elif self.task in consts.TASK_LIST_BINARYCLASS:
            if self.loss == None:
                self.loss = tf.keras.losses.BinaryCrossentropy()
            if self.metrics == None:
                self.metrics = tf.keras.metrics.AUC()
        elif self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
            if self.loss == None:
                self.loss = tf.keras.losses.Huber()
            if self.metrics == None:
                self.metrics = tf.keras.metrics.RootMeanSquaredError()
        else:
            print('Unsupport this task: {}, Apart from [multiclass, binary, \
                    forecast, and regression].'.format(self.task))

        if self.optimizer == consts.OptimizerADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.)
        elif self.optimizer == consts.OptimizerSGD:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, clipnorm=10.)
        else:
            print('Unsupport this optimizer: {}, Default: Adam.'.format(self.optimizer))
            self.optimizer = consts.OptimizerADAM

        if reducelr_patience != 0:
            reduce_lr_callback = ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                                   patience=reducelr_patience, min_lr=0.0001, verbose=1)
            self.callbacks = [reduce_lr_callback]
        if earlystop_patience != 0:
            early_stop_callback = EarlyStopping(monitor=monitor, min_delta=1e-5,
                                                patience=earlystop_patience, verbose=1)
            if self.callbacks is None:
                self.callbacks = [early_stop_callback]
            else:
                self.callbacks.append(early_stop_callback)

    def _dataloader(self, task, X, y, window=1, horizon=1, forecast_length=1, is_train=True):
        """
        Forecast task data format:
        X - 2D DataFrame, shape: (series_length, nb_covariables)
        y - 2D DataFrame, shape: (series_length, nb_target_variables)

        Classification or Regression task data format:
        X - 3D array-like, shape: (nb_samples, series_length, nb_variables)
        y - 2D array-like, shape: (nb_samples, nb_classes)

        window:
        horizon:
        forecast_length:
        Notes
        ----------
        1. The series_length is the timestep for a time series.
        """
        if task in consts.TASK_LIST_FORECAST:
            target_length = y.shape[1]
            continuous_length = len(self.mata.cont_column_names)
            categorical_length = len(self.mata.cat_column_names)
            column_names = self.mata.cont_column_names + self.mata.cat_column_names
            data = pd.concat([y, X], axis=1).drop([self.timestamp], axis=1)
            data = data[column_names].values
            target_start = window - horizon + 1
            inputs = data[:-target_start]
            targets = data[target_start:]
            sequences = from_array_to_timeseries(inputs, targets, forecast_length=forecast_length,
                                                 sequence_length=window)
            X_data, y_data = [], []
            for _, batch in enumerate(sequences):
                X_batch, y_batch = batch
                X_data.append(X_batch.numpy())
                y_data.append(y_batch.numpy())
            X_data = np.concatenate(X_data, axis=0)
            y_data = np.concatenate(y_data, axis=0)[:, :, :target_length]
            if not is_train:
                self.forecast_start = data[-window:].reshape(1, window, data.shape[1])
            if categorical_length != 0:
                X_cont = X_data[:, :, :continuous_length]
                X_cat = X_data[:, :, continuous_length:]
                X_data = [X_cont, X_cat]
                if not is_train:
                    X_cont_start = self.forecast_start[:, :, :continuous_length]
                    X_cat_start = self.forecast_start[:, :, continuous_length:]
                    self.forecast_start = [X_cont_start, X_cat_start]
        else:
            X_data = X
            y_data = y
        return X_data, y_data

    def _preprocessor(self, X, y):
        if isinstance(X.iloc[0, 0], (np.ndarray, pd.Series)):
            self.mata = MetaTSCprocessor()
            X, y = self.mata.fit_transform(X, y)
            self.continuous_columns = self.mata.continuous_columns
            self.categorical_columns = self.mata.categorical_columns
        else:
            self.mata = MetaTSFprocessor(timestamp=self.timestamp, embedding_output_dim=self.embedding_output_dim)
            X, y = self.mata.fit_transform(X, y)
            self.continuous_columns = self.mata.continuous_columns
            self.categorical_columns = self.mata.categorical_columns
        return X, y

    def plot_net(self, file_path):
        plot_model(self.model, to_file=f'{file_path}/model.png', show_shapes=True)

    def save_model(self, file_path, name):
        self.model.save(f'{file_path}/{name}.h5')
        print(f'Model has been saved as {name}.h5')

    def load_model(self, file_path):
        self.model = load_model(file_path)
        print('Model restored')

    def save_model_json(self, file_path, name):
        model_json = self.model.to_json()
        with open(f'{file_path}/{name}.json', 'w') as json_file:
            json_file.write(model_json)
            self.model.save_weights(f'{file_path}/{name}.h5')
        print('Save model to disk.')

    def load_model_json(self, file_path, name):
        with open(f'{file_path}/{name}.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f'{file_path}/{name}.h5')
        print('Loaded model from disk.')