# -*- coding:utf-8 -*-
import os
import time
import math
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from hyperts.framework.dl import layers, losses, metrics
from hyperts.framework.dl.dl_utils.timeseries import from_array_to_timeseries
from hyperts.framework.dl.dl_utils.metainfo import MetaTSFprocessor, MetaTSCprocessor

from hyperts.utils import consts, get_tool_box

from hypernets.utils import logging, fs

logger = logging.get_logger(__name__)

import warnings
warnings.filterwarnings("ignore")


class Metrics(collections.UserDict):
    """A User Dict class to store metrics required for model configuration.
       Usage with `compile()` API.

    Returns
    -------
    A tf.keras.metrics.Metirc object.
    """
    def __init__(self, *args, **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)

        self.data = {
            'mae': metrics.MeanAbsoluteError(name='mae'),
            'mean_absolute_error': metrics.MeanAbsoluteError(name='mae'),
            'mape': metrics.MeanAbsolutePercentageError(name='mape'),
            'mean_absolute_percentage_error': metrics.MeanAbsolutePercentageError(name='mape'),
            'smape': metrics.SymmetricMeanAbsolutePercentageError(name='smape'),
            'mse': metrics.MeanSquaredError(name='mse'),
            'mean_squared_error': metrics.MeanSquaredError(name='mse'),
            'rmse': metrics.RootMeanSquaredError(name='rmse'),
            'msle': metrics.MeanSquaredLogarithmicError(name='msle'),
            'accuracy': metrics.CategoricalAccuracy(name='acc'),
            'auc': metrics.AUC(name='auc'),
            'roc_auc_score': metrics.AUC(name='auc'),
            'precision': metrics.Precision(name='precison'),
            'precision_score': metrics.Precision(name='precison'),
            'recall': metrics.Recall(name='recall'),
            'recall_score': metrics.Recall(name='recall'),
        }

class Losses(collections.UserDict):
    """A User Dict class to store loss required for model configuration.
       Usage with `compile()` API.

    Returns
    -------
    A tf.keras.metrics.Loss object.
    """
    def __init__(self, *args, **kwargs):
        super(Losses, self).__init__(*args, **kwargs)

        self.data = {
            'mse': losses.MeanSquaredError(),
            'mean_squared_error': losses.MeanSquaredError(),
            'mae': losses.MeanAbsoluteError(),
            'mean_absolute_error': losses.MeanAbsoluteError(),
            'huber_loss': losses.Huber(),
            'log_gaussian_loss': losses.LogGaussianLoss(),
            'mape': losses.MeanAbsolutePercentageError(),
            'mean_absolute_percentage_error': losses.MeanAbsolutePercentageError(),
            'categorical_crossentropy': losses.CategoricalCrossentropy(),
            'binary_crossentropy': losses.BinaryCrossentropy(),
        }


class BaseDeepEstimator(object):
    """Abstract base class representing deep estimator object.

    Parameters
    ----------
    task: 'str' or None, default None.
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
        See consts.py for details.
    timestamp: 'str' or None, default None, representing time column name (in DataFrame).
    window: 'int' or None, default None, length of the time series sequences for a sample.
        This must be specified for a forecast task.
    horizon: 'int' or None, default None, representing the time interval between the start point
        of prediction time and the end point of observation time.
        This must be specified for a forecast task.
    forecast_length: 'int', default 1.
        A forecast field of vision during a forecast task.
    monitor_metric: 'str' or None, default None.
        Quantity to be monitored.
    reducelr_patience: 'int' or None, default None.
        Number of epochs with no improvement after which learning rate will be reduced.
    earlystop_patience: 'str' or None, default None.
        Number of epochs with no improvement after which training will be stopped.
    embedding_output_dim: 'int', default 4.
        The dimension in which categorical variables are embedded.
    continuous_columns: CategoricalColumn class.
        Contains some information(name, column_names, input_dim, dtype,
        input_name) about continuous variables.
    categorical_columns: CategoricalColumn class.
        Contains some information(name, vocabulary_size, embedding_dim,
        dtype, input_name) about categorical variables.
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
        self.embedding_output_dim = embedding_output_dim
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.time_columns = None
        self.forecast_start = None
        self.model = None
        self.mata = None

    def _build_estimator(self, **kwargs):
        """Build a time series deep neural net model.

        Returns
        -------
        A tf.keras.Model object.
        """
        raise NotImplementedError(
            '_build_estimator is a protected abstract method, it must be implemented.'
        )

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        """Fit time series model to training data.

        Returns
        -------
        A fitted model and history.
        """
        raise NotImplementedError(
            '_fit is a protected abstract method, it must be implemented.'
        )

    def _predict(self, X):
        """Predict target for sequences in X.

        Returns
        -------
        y: 2D numpy.
        """
        raise NotImplementedError(
            '_predict is a protected abstract method, it must be implemented.'
        )


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
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Parameters
        ----------
        X: 2D DataFrame, shape: (series_length, nb_covariables) for forecast task,
            2D nested DataFrame, shape: (sample, nb_covariables(series_length)) for
            classification or regression task,
        y:  2D DataFrame, shape: (series_length, nb_target_variables) for forecast task,
            2D DataFrame, shape: (nb_samples, 1) for classification or regression task,
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 'auto', 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so verbose=2 is
            recommended when not running interactively (eg, in a production
            environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
            and `tf.keras.callbacks.History` callbacks are created automatically
            and need not be passed into `model.fit`.
            `tf.keras.callbacks.ProgbarLogger` is created or not based on
            `verbose` argument to `model.fit`.
            Callbacks with batch-level calls are currently unsupported with
            `tf.distribute.experimental.ParameterServerStrategy`, and users are
            advised to implement epoch-level calls instead with an appropriate
            `steps_per_epoch` value.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
            `validation_split` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact
            that the validation loss of data provided using `validation_split`
            or `validation_data` is not affected by regularization layers like
            noise and dropout.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
              - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
              - A `tf.data.Dataset`.
              - A Python generator or `keras.utils.Sequence` returning
              `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            `validation_data` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is ignored
            when `x` is a generator or an object of tf.data.Dataset.
            'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample. This
            argument is not supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            When passing an infinitely repeating dataset, you must specify the
            `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
            will run indefinitely with an infinitely repeating dataset.
            This argument is not supported with array inputs.
            When using `tf.distribute.experimental.ParameterServerStrategy`:
              * `steps_per_epoch=None` is not supported.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of an infinitely repeated dataset, it will run into an
            infinite loop. If 'validation_steps' is specified and only part of
            the dataset will be consumed, the evaluation will start from the
            beginning of the dataset at each epoch. This ensures that the same
            validation samples are used every time.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

        See `tf.keras.model.Model.fit` for details.
        """
        start = time.time()
        X, y = self._preprocessor(X, y)
        tb = get_tool_box(X)
        if validation_data is not None:
            validation_data = self.mata.transform(*validation_data)

        if validation_data is None:
            if self.task in consts.TASK_LIST_FORECAST:
                X, X_val, y, y_val = tb.temporal_train_test_split(X, y, test_size=validation_split)
                X = tb.concat_df([X, X_val], axis=0)
                y = tb.concat_df([y, y_val], axis=0)
            else:
                X, X_val, y, y_val = tb.random_train_test_split(X, y, test_size=validation_split)
        else:
            if len(validation_data) != 2:
                raise ValueError(f'Unexpected validation_data length, expected 2 but {len(validation_data)}.')
            X_val, y_val = validation_data[0], validation_data[1]

        if batch_size is None:
            batch_size = min(int(len(X) / 16), 128)
        if steps_per_epoch is None:
            steps_per_epoch = len(X) // batch_size - 1
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

        callbacks = self._inject_callbacks(callbacks, epochs, self.reducelr_patience, self.earlystop_patience)

        model, history = self._fit(X_train, y_train, X_valid, y_valid, epochs=epochs, batch_size=batch_size,
                                   initial_epoch=initial_epoch,
                                   verbose=verbose, callbacks=callbacks, shuffle=shuffle, class_weight=class_weight,
                                   sample_weight=sample_weight,
                                   steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                   validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
                                   use_multiprocessing=use_multiprocessing)
        self.model = model
        logger.info(f'Training finished, total taken {time.time() - start}s.')
        return history

    def predict(self, X, batch_size=128):
        """Inference Function.

        Task: time series classification or regression.
        """
        start = time.time()
        probs = self.predict_proba(X, batch_size)
        preds = self.proba2predict(probs, encode_to_label=True)
        logger.info(f'predict taken {time.time() - start}s')
        return preds

    def forecast(self, X):
        """Inference Function.

        Task: time series forecast.
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
            X = X[X_cont_cols + X_cat_cols].values.astype(consts.DATATYPE_TENSOR_FLOAT)

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
        """Inference Function.

        Task: time series classification/regression.
        """
        probs = []
        X = self.mata.transform_X(X)
        sample_size, iters = X.shape[0], X.shape[0] // batch_size + 1
        for idx in range(iters):
            proba = self._predict(X[idx * batch_size:min((idx + 1) * batch_size, sample_size)])
            probs.append(proba.numpy())
        probs = np.concatenate(probs, axis=0)
        if probs.shape[-1] == 1 and self.task in consts.TASK_LIST_CLASSIFICATION:
            probs = np.hstack([1 - probs, probs])
        return probs

    def proba2predict(self, proba, encode_to_label=True):
        """Transition Function.

        Task: time series classification.
        """
        if self.task in consts.TASK_LIST_REGRESSION:
            return proba
        if proba is None:
            raise ValueError('[proba] can not be none.')
        if len(proba.shape) == 1:
            proba = proba.reshape((-1, 1))
        if proba.shape[-1] > 2:
            predict = np.zeros(shape=proba.shape)
            argmax = proba.argmax(axis=-1)
            predict[np.arange(len(argmax)), argmax] = 1
        elif proba.shape[-1] == 2:
            predict = proba.argmax(axis=-1)
        else:
            predict = (proba > 0.5).astype('int32').reshape((-1, 1))

        if encode_to_label:
            logger.info('reverse indicators to labels.')
            predict = self.mata.inverse_transform_y(predict)
        return predict

    def _inject_callbacks(self, callbacks, epochs, reducelr_patience=5, earlystop_patience=10, verbose=1):
        """Inject callbacks. including ReduceLROnPlateau and EarlyStopping.

        Parameters
        ----------
        reducelr_patience: 'int' or None, default None.
            Number of epochs with no improvement after which learning rate will be reduced.
        earlystop_patience: 'str' or None, default None.
            Number of epochs with no improvement after which training will be stopped.
        verbose: 'int'. 0: quiet, 1: update messages.
        """

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
        """Configures the model for training.

        Parameters
        ----------
        model: `Model` groups layers into an object with training and
            inference features. See `tf.keras.models.Model`.
        optimizer: String (name of optimizer). See `tf.keras.optimizers`.
        learning_rate: 'float', default 0.001.
        """

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
            logger.info('Unsupport this task: {}, Apart from [multiclass, binary, \
                         forecast, and regression].'.format(self.task))

        loss = Losses()[self.loss]
        if set(self.metrics) < set(Metrics().keys()):
            metrics = [Metrics()[m] for m in self.metrics]
        else:
            if self.task in consts.TASK_LIST_BINARYCLASS:
                metrics = [Metrics()['auc']]
            elif self.task in consts.TASK_LIST_MULTICLASS:
                metrics = [Metrics()['accuracy']]
            else:
                metrics = [Metrics()['rmse']]
            logger.warning(f"In dl model, {self.metrics} is not supported, "
                           f"so ['{metrics[0].name}'] will be called.")

        if optimizer == 'auto':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.)
            logger.info("The optimizer is 'auto', default: Adam, learning rate=0.001.")
        elif optimizer == consts.OptimizerADAM:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=10.)
        elif optimizer == consts.OptimizerSGD:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=10.)
        else:
            raise ValueError(f'Unsupport this optimizer: [optimizer].')

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def _dataloader(self, task, X, y, window=1, horizon=1, forecast_length=1, is_train=True):
        """ Load data set.

        Parameters
        ----------
        Forecast task data format:
        X - 2D DataFrame, shape: (series_length, nb_covariables)
        y - 2D DataFrame, shape: (series_length, nb_target_variables)

        Classification or Regression task data format:
        X - 3D array-like, shape: (nb_samples, series_length, nb_variables)
        y - 2D array-like, shape: (nb_samples, nb_classes)

        window: 'int' or None, default None, length of the time series sequences for a sample.
            This must be specified for a forecast task.
        horizon: 'int' or None, default None, representing the time interval between the start point
            of prediction time and the end point of observation time.
            This must be specified for a forecast task.
        forecast_length: 'int', default 1.
            A forecast field of vision during a forecast task.
        """

        if task in consts.TASK_LIST_FORECAST:
            tb = get_tool_box(X, y)
            target_length = tb.get_shape(y)[1]
            continuous_length = len(self.mata.cont_column_names)
            categorical_length = len(self.mata.cat_column_names)
            column_names = self.mata.cont_column_names + self.mata.cat_column_names
            data = tb.concat_df([y, X], axis=1).drop([self.timestamp], axis=1)
            data = tb.df_to_array(data[column_names]).astype(consts.DATATYPE_TENSOR_FLOAT)
            target_start = window - horizon + 1
            inputs = data[:-target_start]
            targets = data[target_start:]
            sequences = from_array_to_timeseries(inputs, targets, forecast_length=forecast_length, sequence_length=window)
            X_data, y_data = [], []
            for _, batch in enumerate(sequences):
                X_batch, y_batch = batch
                X_data.append(X_batch.numpy())
                y_data.append(y_batch.numpy())
            try:
                X_data = np.concatenate(X_data, axis=0)
                y_data = np.concatenate(y_data, axis=0)[:, :, :target_length]
            except:
                raise ValueError(f'Reset forecast_window, which should be less than {X//2}.')
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
                tb = get_tool_box(X)
                self.window = tb.get_shape(X)[1]
            X_data = X.astype('float32')
            y_data = y
        return X_data, y_data

    def _from_tensor_slices(self, X, y, batch_size, epochs=None, shuffle=False, drop_remainder=True):
        """Creates a `Dataset` whose elements are slices of the given tensors.

        Returns
        ----------
        Dataset: A `Dataset`.
        """
        data = {}
        for c in self.continuous_columns:
            if isinstance(X, list):
                data[c.name] = X[0].astype(consts.DATATYPE_TENSOR_FLOAT)
            else:
                data[c.name] = X.astype(consts.DATATYPE_TENSOR_FLOAT)

        if self.categorical_columns is not None and len(self.categorical_columns) > 0:
            data['input_categorical_vars_all'] = X[1].astype(consts.DATATYPE_TENSOR_FLOAT)

        dataset = tf.data.Dataset.from_tensor_slices((data, y))

        if shuffle:
            dataset = dataset.shuffle(y.shape[0])

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder and y.shape[0] >= batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if epochs is not None:
            dataset = dataset.repeat(epochs+1)

        return dataset

    def _preprocessor(self, X, y):
        """ The feature is preprocessed and the continuous columns and categorical columns
            related information is obtained.

        Notes
        ----------
        continuous_columns: CategoricalColumn class.
            Contains some information(name, column_names, input_dim, dtype,
            input_name) about continuous variables.
        categorical_columns: CategoricalColumn class.
            Contains some information(name, vocabulary_size, embedding_dim,
            dtype, input_name) about categorical variables.
        """
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
        """Gets monitor for ReduceLROnPlateau and EarlyStopping.

        """
        monitor = self.monitor_metric
        if monitor is None:
            if self.metrics is not None and len(self.metrics) > 0:
                monitor = 'val_' + self.first_metric_name
        return monitor

    @property
    def first_metric_name(self):
        """Get first metric name.

        """
        if self.metrics is None or len(self.metrics) <= 0:
            raise ValueError('`metrics` is none or empty.')
        first_metric = self.metrics[0]
        if isinstance(first_metric, str):
            return first_metric
        if callable(first_metric):
            return first_metric.__name__
        raise ValueError('`metric` must be string or callable object.')

    def plot_net(self, model_file):
        """Plot net model architecture.

        """
        plot_model(self.model, to_file=f'{model_file}/model.png', show_shapes=True)

    def save_model(self, model_file, name='dl_model'):
        """Save the instance object.

        """
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
        """Load the instance object.

        """
        import h5py, io
        try:
            from tensorflow.python import keras
            from hyperts.framework.dl.dl_utils.saveconfig import compile_args_from_training_config
            keras.saving.saving_utils.compile_args_from_training_config = compile_args_from_training_config
        except:
            raise ValueError('Perhaps updating version Tensorflow above 2.3.0 will solve the issue.')

        custom_objects = {}
        custom_objects.update(layers.layers_custom_objects)
        custom_objects.update(losses.losses_custom_objects)
        custom_objects.update(metrics.metrics_custom_objects)

        if model_file.endswith('.pkl'):
            model_file = os.path.splitext(model_file)[0]
        with fs.open(f'{model_file}_{name}.h5', "rb") as fp:
            data = fp.read()
        buf = io.BytesIO(data)
        del data
        with h5py.File(buf, 'r') as h:
            model = load_model(h, custom_objects=custom_objects)
        logger.info('Loaded model from disk.')
        return model