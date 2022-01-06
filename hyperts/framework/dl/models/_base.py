# -*- coding:utf-8 -*-
import os
import time
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from hyperts.framework.dl import layers
from hyperts.framework.dl.timeseries import from_array_to_timeseries
from hyperts.framework.dl.metainfo import MetaTSFprocessor, MetaTSCprocessor

from hyperts.utils import consts, get_tool_box

from hypernets.utils import logging, fs

logger = logging.get_logger(__name__)

import warnings
warnings.filterwarnings("ignore")

Metircs = {
    'mse': tf.keras.metrics.MeanSquaredError(name='mse'),
    'rmse': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
    'accuracy': tf.keras.metrics.CategoricalAccuracy(name='acc'),
    'auc': tf.keras.metrics.AUC(name='auc'),
    'precison': tf.keras.metrics.Precision(name='precison'),
    'recall': tf.keras.metrics.Recall(name='precall'),
}

Losses = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mean_squared_error': tf.keras.losses.MeanSquaredError(),
    'mae': tf.keras.losses.MeanAbsoluteError(),
    'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
    'huber_loss': tf.keras.losses.Huber(),
    'mape': tf.keras.losses.MeanAbsolutePercentageError(),
    'mean_absolute_percentage_error': tf.keras.losses.MeanAbsolutePercentageError(),
    'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),

    'log_gaussian_loss': layers.log_gaussian_loss,
}


class BaseDeepEstimator(object):
    """

    """

    def __init__(self, task,
                 timestamp=None,
                 window=None,
                 horizon=None,
                 forecast_length=1,
                 monitor_metric=None,
                 reducelr_patience=None,
                 earlystop_patience=None,
                 embedding_output_dim=4,
                 continuous_columns=None,
                 categorical_columns=None):
        self.task = task
        self.timestamp = timestamp
        self.window = window
        self.horizon = horizon
        self.forecast_length = forecast_length
        self.monitor_metric = monitor_metric
        self.reducelr_patience = reducelr_patience
        self.earlystop_patience = earlystop_patience
        self.embedding_output_dim=embedding_output_dim
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.time_columns = None
        self.forecast_start = None
        self.model = None

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
        tb = get_tool_box(X)
        if validation_data is not None:
            validation_data = self.mata.transform(*validation_data)

        if validation_data is None:
            if self.task in consts.TASK_LIST_FORECAST:
                X, X_val, y, y_val = tb.temporal_train_test_split(X, y, test_size=validation_split)
            else:
                X, X_val, y, y_val = tb.random_train_test_split(X, y, test_size=validation_split)
        else:
            if len(validation_data) != 2:
                raise ValueError(f'Unexpected validation_data length, expected 2 but {len(validation_data)}.')
            X_val, y_val = validation_data[0], validation_data[1]

        if batch_size is None:
            batch_size = min(int(len(X) / 16), 128)

        X_train, y_train = self._dataloader(self.task, X, y, self.window, self.horizon, self.forecast_length,
                                            is_train=True)
        X_valid, y_valid = self._dataloader(self.task, X_val, y_val, self.window, self.horizon, self.forecast_length,
                                            is_train=False)

        callbacks = self._inject_callbacks(callbacks, epochs, self.reducelr_patience, self.earlystop_patience)

        model, history = self._fit(X_train, y_train, X_valid, y_valid, epochs=epochs, batch_size=batch_size,
                            initial_epoch=initial_epoch,
                            verbose=verbose, callbacks=callbacks, shuffle=shuffle, class_weight=class_weight,
                            sample_weight=sample_weight,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                            validation_batch_size=validation_batch_size,
                            validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
                            use_multiprocessing=use_multiprocessing)
        self.model = model
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
        data = self.forecast_start.copy()
        if X.shape[1] >= 1:
            continuous_length = len(self.mata.cont_column_names)
            categorical_length = len(self.mata.cat_column_names)
            for i in range(math.ceil(steps / self.forecast_length)):
                pred = self._predict(data)
                futures.append(pred.numpy())
                covariable = np.expand_dims(X[i:i + self.forecast_length], 0)
                forcast = np.concatenate([pred.numpy(), covariable], axis=-1)
                if categorical_length > 0:
                    data[0] = np.append(data[0], forcast[:, :, :continuous_length]).reshape((1, -1, continuous_length))
                    data[1] = np.append(data[1], forcast[:, :, continuous_length:]).reshape((1, -1, categorical_length))
                    data = [data[0][:, -self.window:, :], data[1][:, -self.window:, :]]
                else:
                    data = np.append(data, forcast).reshape((1, -1, continuous_length))
                    data = data[:, -self.window:, :]
        else:
            for i in range(math.ceil(steps / self.forecast_length)):
                pred = self._predict(data)
                futures.append(pred.numpy())
                forcast = pred.numpy()
                data = np.append(data, forcast).reshape((1, -1, self.mata.classes_))
                data = data[:, -self.window:, :]

        logger.info(f'forecast taken {time.time() - start}s')
        return np.array(futures).reshape(steps, -1)

    def predict_proba(self, X, batch_size=128):
        probs = []
        X = self.mata.transform_X(X)
        sample_size, iters = X.shape[0], X.shape[0] // batch_size + 1
        for idx in range(iters):
            proba = self._predict(X[idx*batch_size:min((idx+1)*batch_size, sample_size)])
            probs.append(proba.numpy())
        probs = np.concatenate(probs, axis=0)
        if probs.shape[-1] == 1:
            probs = np.hstack([1 - probs, probs])
        return probs

    def proba2predict(self, proba, encode_to_label=True):
        if self.task in consts.TASK_LIST_REGRESSION:
            return proba
        if proba is None:
            raise ValueError('[proba] can not be none.')
        if len(proba.shape) == 1:
            proba = proba.reshape((-1, 1))
        if proba.shape[-1] > 1:
            predict = np.zeros(shape=proba.shape)
            argmax = proba.argmax(axis=-1)
            predict[np.arange(len(argmax)), argmax] = 1
        else:
            predict = (proba > 0.5).astype('int32').reshape((-1, 1))

        if encode_to_label:
            logger.info('reverse indicators to labels.')
            predict = self.mata.inverse_transform_y(predict)
        return predict

    def _inject_callbacks(self, callbacks, epochs, reducelr_patience=5, earlystop_patience=10, verbose=1):
        lr, es = None, None
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, ReduceLROnPlateau):
                    lr = callback
                if isinstance(callback, EarlyStopping):
                    es = callback
        else:
            callbacks = []

        if epochs <= 10:
            return []
        else:
            if lr is None and isinstance(reducelr_patience, int) and reducelr_patience > 0:
                lr = ReduceLROnPlateau(monitor=self.monitor, factor=0.5,
                        patience=reducelr_patience, min_lr=0.0001, verbose=verbose)
                callbacks.append(lr)
                logger.info(f'Injected a callback [ReduceLROnPlateau]. monitor:{lr.monitor}, '
                            f'patience:{lr.patience}')
            if es is None and isinstance(earlystop_patience, int) and earlystop_patience > 0:
                es = EarlyStopping(monitor=self.monitor, min_delta=1e-5,
                        patience=earlystop_patience, verbose=verbose)
                callbacks.append(es)
                logger.info(f'Injected a callback [EarlyStopping]. monitor:{es.monitor}, '
                            f'patience:{es.patience}')
            return callbacks


    def _compile_model(self, model, optimizer, learning_rate=0.001):
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
            if self.loss == 'auto':
                self.loss = 'huber_loss'
            if self.metrics == 'auto':
                self.metrics = ['rmse']
        elif self.task in consts.TASK_LIST_MULTICLASS:
            if self.loss == 'auto':
                self.loss = 'categorical_crossentropy'
            if self.metrics == 'auto':
                self.metrics = ['accuracy']
        elif self.task in consts.TASK_LIST_BINARYCLASS:
            if self.loss == 'auto':
                self.loss = 'binary_crossentropy'
            if self.metrics == 'auto':
                self.metrics = ['auc']
        else:
            print('Unsupport this task: {}, Apart from [multiclass, binary, \
                    forecast, and regression].'.format(self.task))
        loss = Losses[self.loss]
        metrics = [Metircs[m] for m in self.metrics]

        if optimizer == 'auto':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.)
            print("The optimizer is 'auto', default: Adam, learning rate=0.001.")
        elif optimizer == consts.OptimizerADAM:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.)
        elif optimizer == consts.OptimizerSGD:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=10.)
        else:
            raise ValueError(f'Unsupport this optimizer: [optimizer].')

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

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
            if is_train:
                self.window = X.shape[1]
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

    @property
    def monitor(self):
        monitor = self.monitor_metric
        if monitor is None:
            if self.metrics is not None and len(self.metrics) > 0:
                monitor = 'val_' + self.first_metric_name
        return monitor

    @property
    def first_metric_name(self):
        if self.metrics is None or len(self.metrics) <= 0:
            raise ValueError('`metrics` is none or empty.')
        first_metric = self.metrics[0]
        if isinstance(first_metric, str):
            return first_metric
        if callable(first_metric):
            return first_metric.__name__
        raise ValueError('`metric` must be string or callable object.')

    def plot_net(self, model_file):
        plot_model(self.model, to_file=f'{model_file}/model.png', show_shapes=True)

    def save_model(self, model_file, name='dl_model'):
        import h5py, io
        if model_file.endswith('.pkl'):
            model_file = os.path.splitext(model_file)[0]
        with fs.open(f'{model_file}_{name}.h5', "wb") as fw:
            buf = io.BytesIO()
            with h5py.File(buf, 'w') as h:
                save_model(self.model, h, save_format='h5')
            data = buf.getvalue()
            buf.close()
            fw.write(data)
        self.model = None
        logger.info('Save model to disk.')

    @staticmethod
    def load_model(model_file, name='dl_model'):
        import h5py, io
        if model_file.endswith('.pkl'):
            model_file = os.path.splitext(model_file)[0]
        with fs.open(f'{model_file}_{name}.h5', "rb") as fp:
            data = fp.read()
        buf = io.BytesIO(data)
        del data
        with h5py.File(buf, 'r') as h:
            model = load_model(h, custom_objects=layers.custom_objects)
        logger.info('Loaded model from disk.')
        return model