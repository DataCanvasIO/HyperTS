# -*- coding:utf-8 -*-
"""

"""
import copy
import numpy as np

from hypernets.core import set_random_state
from hypernets.experiment.compete import SteppedExperiment, ExperimentStep, \
                                         SpaceSearchStep, EnsembleStep, FinalTrainStep

from hypernets.utils import logging

from hyperts.utils import consts, get_tool_box
from hyperts.toolbox import generate_anomaly_pseudo_ground_truth
from hyperts.utils.plot import plot_plotly, plot_mpl, enable_plotly, enable_mpl

logger = logging.get_logger(__name__)


def _set_log_level(log_level):
    logging.set_level(log_level)


class TSAFDataPreprocessStep(ExperimentStep):
    """Time Series Anomaly Detection or Forecasting Task Data Preprocess Step.

    """

    def __init__(self, experiment, name, timestamp_col=None, freq=None,
                 cv=False, covariate_cols=None, covariate_cleaner=None,
                 ensemble_size=None, train_data_periods=None,
                 anomaly_label_col=None, contamination=0.05):
        super().__init__(experiment, name)

        timestamp_col = [timestamp_col] if isinstance(timestamp_col, str) else timestamp_col
        covariate_cols = [covariate_cols] if isinstance(covariate_cols, str) else covariate_cols

        self.cv = cv
        self.ensemble_size = ensemble_size

        self.freq = freq
        self.target_cols = None
        self.history_prior = None
        self.covariate_cols = covariate_cols
        self.covariate_cleaner = covariate_cleaner
        self.train_data_periods = train_data_periods
        self.anomaly_label_col = anomaly_label_col
        self.contamination = contamination
        self.timestamp_col = timestamp_col if timestamp_col is not None else consts.TIMESTAMP

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)

        tb = get_tool_box(X_train, y_train)

        if self.cv and X_eval is not None and y_eval is not None:
            logger.info(f'{self.name} cv enabled, so concat train data and eval data')
            X_train = tb.concat_df([X_train, X_eval], axis=0)
            y_train = tb.concat_df([y_train, y_eval], axis=0)
            X_eval = None
            y_eval = None

        # 1. covariates data clean procsss
        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            X_train = self.covariate_transform(X_train)
        self.step_progress('transform covariate variables')

        # 2. target plus covariates process
        train_Xy = tb.concat_df([X_train, y_train], axis=1)
        variable_cols = tb.list_diff(train_Xy.columns, self.timestamp_col)
        target_cols = tb.list_diff(variable_cols, self.covariate_cols)
        excluded_cols = tb.list_diff(train_Xy.columns, target_cols)
        train_Xy = self.series_transform(train_Xy, target_cols)
        X_train, y_train = train_Xy[excluded_cols], train_Xy[target_cols]
        self.step_progress('fit_transform train set')

        self.history_prior = tb.df_mean_std(y_train)

        # 3. eval variables data process
        if not self.cv:
            if X_eval is None or y_eval is None:
                if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
                    period = tb.fft_infer_period(y_train.iloc[:, 0])
                    if isinstance(self.experiment.eval_size, int):
                        eval_horizon = self.experiment.eval_size
                    elif int(X_train.shape[0] * consts.DEFAULT_MIN_EVAL_SIZE) <= period:
                        eval_horizon = period
                    else:
                        eval_horizon = consts.DEFAULT_MIN_EVAL_SIZE
                    X_train, X_eval, y_train, y_eval = \
                        tb.temporal_train_test_split(X_train, y_train, test_size=eval_horizon)
                    self.step_progress('split into train set and eval set')
            else:
                if self.covariate_cols is not None and len(self.covariate_cols) > 0:
                    X_eval = self.covariate_transform(X_eval)
                eval_Xy = tb.concat_df([X_eval, y_eval], axis=1)
                eval_Xy = self.series_transform(eval_Xy, target_cols)
                X_eval, y_eval = eval_Xy[excluded_cols], eval_Xy[target_cols]
                self.step_progress('transform eval set')

        if self.task in consts.TASK_LIST_DETECTION:
            if self.anomaly_label_col is None:
                train_y, eval_y = generate_anomaly_pseudo_ground_truth(
                                  X_train=y_train,
                                  X_test=y_eval,
                                  contamination=self.contamination,
                                  random_state=9527)
                anomaly_label_col = consts.TARGET
            else:
                train_y = y_train.pop(self.anomaly_label_col)
                eval_y = y_eval.pop(self.anomaly_label_col)
                anomaly_label_col = self.anomaly_label_col

            X_train = tb.concat_df([X_train, y_train], axis=1)
            y_train = tb.DataFrame(train_y, columns=[anomaly_label_col])
            if X_eval is not None or y_eval is not None:
                X_eval = tb.reset_index(X_eval)
                y_eval = tb.reset_index(y_eval)
                pos_label = tb.infer_pos_label(y_train, self.task, anomaly_label_col)
                if pos_label not in eval_y:
                    n, m = X_eval.shape[0], X_eval.shape[1]
                    outliers_indices = np.random.randint(n, size=int(n * 0.05))
                    features_indices = np.random.randint(m, size=3 if m > 3 else 1)
                    for i in outliers_indices:
                        for j in features_indices:
                            scale = y_eval.iloc[i, j]
                            noise = np.random.normal(0, scale, size=1)
                            y_eval.iloc[i, j] = y_eval.iloc[i, j] * 3 + noise
                    eval_y[outliers_indices] = pos_label
                X_eval = tb.concat_df([X_eval, y_eval], axis=1)
                y_eval = tb.DataFrame(eval_y, columns=[anomaly_label_col])
        else:
            anomaly_label_col = None

        # 4. compute new data shape
        data_shapes = {'X_train.shape': tb.get_shape(X_train),
                       'y_train.shape': tb.get_shape(y_train),
                       'X_eval.shape': tb.get_shape(X_eval, allow_none=True),
                       'y_eval.shape': tb.get_shape(y_eval, allow_none=True),
                       'X_test.shape': tb.get_shape(X_test, allow_none=True)
                       }

        # 5. reset part parameters
        self.data_shapes = data_shapes
        if self.task in consts.TASK_LIST_DETECTION:
            self.target_cols = anomaly_label_col
        else:
            self.target_cols = target_cols
        self.indicator_cols_ = target_cols

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        if self.task in consts.TASK_LIST_DETECTION:
            target_cols = self.indicator_cols_
        else:
            target_cols = None
        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            X_transform = self.covariate_transform(X)
            X_transform = self.series_transform(X_transform, target_cols)
        else:
            X_transform = self.series_transform(X, target_cols)

        return X_transform

    def covariate_transform(self, X):
        X = copy.deepcopy(X)
        X = self.covariate_cleaner.transform(X)
        return X

    def series_transform(self, X, target_cols=None):
        X = copy.deepcopy(X)
        tb = get_tool_box(X)
        covar_object_names, covar_float_names = [], []

        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            for col in self.covariate_cols:
                if X[col].dtypes == consts.DataType_OBJECT:
                    covar_object_names.append(col)
                elif X[col].dtypes == consts.DataType_FLOAT:
                    covar_float_names.append(col)

        if target_cols is not None:
            impute_col_names = target_cols + covar_float_names
        else:
            impute_col_names = covar_float_names

        X = tb.sort_values(X, ts_name=self.timestamp_col[0])
        self.freq = self.freq if self.freq is not None else \
            tb.infer_ts_freq(X[self.timestamp_col], ts_name=self.timestamp_col[0])
        X = tb.drop_duplicated_ts_rows(X, ts_name=self.timestamp_col[0])
        X = tb.smooth_missed_ts_rows(X, freq=self.freq, ts_name=self.timestamp_col[0])

        # if target_cols is not None and len(target_cols) > 0:
        #     X[target_cols] = tb.nan_to_outliers(X[target_cols])
        if impute_col_names is not None and len(impute_col_names) > 0:
            X[impute_col_names] = tb.multi_period_loop_imputer(X[impute_col_names], freq=self.freq)
        if covar_object_names is not None and len(covar_object_names) > 0:
            X[covar_object_names] = X[covar_object_names].fillna(method='ffill').fillna(method='bfill')

        if isinstance(self.train_data_periods, int) and self.train_data_periods < len(X):
            X = tb.select_1d_reverse(X, self.train_data_periods)
            X = tb.reset_index(X)
            logger.info(f'Cut off training data, and the retention period length is {self.train_data_periods}.')

        return X

    def get_params(self, deep=True):
        params = super(TSAFDataPreprocessStep, self).get_params()
        return params

    def get_fitted_params(self):
        freq = self.freq if self.freq is not None else None
        params = super().get_fitted_params()
        data_shapes = self.data_shapes if self.data_shapes is not None else {}
        return {**params, **data_shapes, 'freq': freq}


class TSRCDataPreprocessStep(ExperimentStep):
    """Time Series Classification or Regression Task Data Preprocess Step.

    """

    def __init__(self, experiment, name, cv=False):
        super().__init__(experiment, name)

        self.cv = cv

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)

        tb = get_tool_box()

        if self.cv and X_eval is not None and y_eval is not None:
            logger.info(f'{self.name} cv enabled, so concat train data and eval data')
            X_train = tb.concat_df([X_train, X_eval], axis=0)
            y_train = tb.concat_df([y_train, y_eval], axis=0)
            X_eval = None
            y_eval = None

        # 1. data clean procsss
        X_train, y_train = self.panel_transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        # 2. eval data process
        if not self.cv:
            if X_eval is None and y_eval is None:
                eval_size = self.experiment.eval_size
                if self.task == consts.Task_REGRESSION:
                    X_train, X_eval, y_train, y_eval = \
                        tb.train_test_split(X_train, y_train, test_size=eval_size,
                                            random_state=self.experiment.random_state)
                else:
                    X_train, X_eval, y_train, y_eval = \
                        tb.train_test_split(X_train, y_train, test_size=eval_size,
                                            random_state=self.experiment.random_state, stratify=y_train)
            else:
                X_eval, y_eval = self.panel_transform(X_eval, y_eval)
                self.step_progress('transform eval set')

        # 3. compute new data shape
        data_shapes = {'X_train.shape': tb.get_shape(X_train),
                       'y_train.shape': tb.get_shape(y_train),
                       'X_eval.shape': tb.get_shape(X_eval, allow_none=True),
                       'y_eval.shape': tb.get_shape(y_eval, allow_none=True),
                       'X_test.shape': tb.get_shape(X_test, allow_none=True)
                       }

        self.data_shapes_ = data_shapes

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return self.panel_transform(X, y)

    def panel_transform(self, X, y=None):
        y_name = '__tabular-toolbox__Y__'
        X = copy.deepcopy(X)
        if y is not None:
            y = copy.deepcopy(y)
        if y is not None:
            X[y_name] = y

        if y is not None:
            logger.debug('clean the rows which label is NaN')
            X = X.dropna(subset=[y_name])
            y = X.pop(y_name)

        if y is None:
            return X
        else:
            return X, y

    def get_params(self, deep=True):
        params = super(TSRCDataPreprocessStep, self).get_params()
        return params

    def get_fitted_params(self):
        params = super().get_fitted_params()
        data_shapes = self.data_shapes_ if self.data_shapes_ is not None else {}

        return {**params, **data_shapes}


class TSSpaceSearchStep(SpaceSearchStep):
    """Time Series Space Searching.

    """
    def __init__(self, experiment, name, cv=False, num_folds=3):
        super().__init__(experiment, name, cv=cv, num_folds=num_folds)

    def search(self, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        model = copy.deepcopy(self.experiment.hyper_model)
        es = self.find_early_stopping_callback(model.callbacks)
        if es is not None and es.time_limit is not None and es.time_limit > 0:
            es.time_limit = self.estimate_time_limit(es.time_limit)
        model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds, **kwargs)
        return model


class TSEnsembleStep(EnsembleStep):
    """Time Series Ensemble.

    """
    def __init__(self, experiment, name, mode=None, scorer=None, ensemble_size=10, cv=False, retrain_on_wholedata=False):
        super().__init__(experiment, name, scorer=scorer, ensemble_size=ensemble_size)
        self.mode = mode
        self.cv = cv
        self.retrain_on_wholedata = retrain_on_wholedata

    def build_estimator(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        trials = self.select_trials(hyper_model)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in trials]
        ensemble = self.get_ensemble(estimators, X_train, y_train)

        if self.task not in consts.TASK_LIST_FORECAST and \
            all(['oof' in trial.memo.keys() for trial in trials]):
            logger.info('ensemble with oofs')
            oofs = self.get_ensemble_predictions(trials, ensemble)
            assert oofs is not None
            if hasattr(oofs, 'shape'):
                tb = get_tool_box(y_train, oofs)
                y_, oofs_ = tb.select_valid_oof(y_train, oofs)
                ensemble.fit(None, y_, oofs_)
            else:
                ensemble.fit(None, y_train, oofs)
        elif self.task in consts.TASK_LIST_FORECAST and self.cv:
            tb = get_tool_box(X_train, y_train)
            period = tb.fft_infer_period(y_train.iloc[:, 0])
            if isinstance(self.experiment.eval_size, int):
                eval_horizon = self.experiment.eval_size
            elif int(X_train.shape[0] * consts.DEFAULT_MIN_EVAL_SIZE) <= period:
                eval_horizon = period
            else:
                eval_horizon = consts.DEFAULT_MIN_EVAL_SIZE
            X_train, X_eval, y_train, y_eval = \
            tb.temporal_train_test_split(X_train, y_train, test_size=eval_horizon)
            estimators = self.est_retrain(trials, hyper_model, X_train, y_train, **kwargs)
            # self.retrain_on_wholedata = False
            ensemble.estimators = list(estimators)
            ensemble.classes_ = None
            ensemble.fit(X_eval, y_eval)
        elif X_eval is not None and y_eval is not None:
            ensemble.fit(X_eval, y_eval)
        else:
            ensemble.fit(X_train, y_train)

        if not isinstance(ensemble.weights_, list):
            weights = np.sum(ensemble.weights_, axis=1)
        else:
            weights = ensemble.weights_

        if self.retrain_on_wholedata:
            estimators = self.est_retrain(trials, hyper_model, X_train, y_train, X_eval, y_eval, weights, **kwargs)
        else:
            estimators = np.where(np.array(weights) > 0, estimators, None)

        ensemble.estimators = list(estimators)

        return ensemble

    def get_ensemble(self, estimators, X_train, y_train):
        tb = get_tool_box(X_train, y_train)
        target_dims = tb.get_shape(tb.DataFrame(y_train))[1]
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_REGRESSION:
            ensemble_task = 'regression'
        elif 'binary' in self.task or self.task in consts.TASK_LIST_DETECTION:
            ensemble_task = 'binary'
        else:
            ensemble_task = 'multiclass'
        return tb.greedy_ensemble(ensemble_task, estimators, target_dims=target_dims,
                               scoring=self.scorer, ensemble_size=self.ensemble_size)

    def est_retrain(self, trials, hyper_model, X_train, y_train, X_eval=None, y_eval=None, weights=None, **kwargs):
        estimators = []
        if X_eval is not None or y_eval is not None:
            tb = get_tool_box(X_train, X_eval)
            X_all = tb.concat_df([X_train, X_eval], axis=0)
            y_all = tb.concat_df([y_train, y_eval], axis=0)
        else:
            X_all, y_all = X_train, y_train

        if self.mode is not consts.Mode_STATS:
            kwargs.update({'epochs': kwargs.pop('final_train_epochs', consts.FINAL_TRAINING_EPOCHS)})

        logger.info('Retrain the best trial model with all data ...')
        weights = [1]*len(trials) if weights is None else weights
        for trial, weight in zip(trials, weights):
            if weight > 0:
                estimator = hyper_model.final_train(trial.space_sample, X_all, y_all, **kwargs)
                estimators.append(estimator)
            else:
                estimators.append(None)
        return estimators


class TSFinalTrainStep(FinalTrainStep):
    def __init__(self, experiment, name, mode=None, retrain_on_wholedata=False):
        super().__init__(experiment, name)
        self.mode = mode
        self.retrain_on_wholedata = retrain_on_wholedata

    def build_estimator(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if self.retrain_on_wholedata:
            trial = hyper_model.get_best_trial()
            if X_eval is not None or y_eval is not None:
                tb = get_tool_box(X_train, X_eval)
                X_all = tb.concat_df([X_train, X_eval], axis=0)
                y_all = tb.concat_df([y_train, y_eval], axis=0)
            else:
                X_all, y_all = X_train, y_train

            if self.mode is not consts.Mode_STATS:
                kwargs.update({'epochs': kwargs.pop('final_train_epochs', consts.FINAL_TRAINING_EPOCHS)})

            logger.info('Retrain the best trial model with all data ...')
            estimator = hyper_model.final_train(trial.space_sample, X_all, y_all, **kwargs)
        else:
            estimator = hyper_model.load_estimator(hyper_model.get_best_trial().model_file)

        return estimator


class TSPipeline:
    """Pipeline Extension for Time Series Analysis.

        Parameters
        ----------
        sk_pipeline: sklearn pipeline, including data_preprocessing, space_searching, final_training and estimator
            steps and so on.
        freq: 'str', DateOffset or None, default None.
        task: 'str' or None, default None.
            Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
            See consts.py for details.
        mode: str, default 'stats'. Optional {'stats', 'dl', 'nas'}, where,
            'stats' indicates that all the models selected in the execution experiment are statistical models.
            'dl' indicates that all the models selected in the execution experiment are deep learning models.
            'nas' indicates that the selected model of the execution experiment will be a deep network model
            for neural architecture search, which is not currently supported.
        timestamp: str, forecast task 'timestamp' cannot be None, (default=None).
        covariables: list[n*str], if the data contains covariables, specify the covariable column names, (default=None).
        target: str or list, optional.
            Target feature name for training, which must be one of the train_data columns for classification[str],
            regression[str] or unvariate forecast task [list]. For multivariate forecast task, it is multiple columns
            of training data.
    """

    def __init__(self, sk_pipeline, freq, task, mode, timestamp, covariables, target, history=None, **kwargs):
        self.freq = freq
        self.task = task
        self.mode = mode
        self.target = target
        self.timestamp = timestamp
        self.covariables = covariables

        self.sk_pipeline = sk_pipeline
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
            self.prior = sk_pipeline.named_steps.data_preprocessing.history_prior
            self.history = history

        self.kwargs = kwargs.copy()

    def predict(self, X, forecast_start=None):
        """Predicts target for sequences in X.

        Parameters
        ----------
        X: 'DataFrame'.
            For forecast task, X.columns = ['timestamp', (covariate_1), (covariate_2),...].
            (covariate_1) indicates that it may not exist.
            For classification or regression tasks, X.columns = [variate_1, variate_2,...].
        forecast_start : 'DataFrame'. This parameter applies only to 'dl' mode.
            Forecast the start fragment, if None, by default the last window fragment of the
            train data.
            forecast_start.columns = ['timestamp', variate_1, variate_2, ..., (covariate_1), (covariate_2),...].
            (covariate_1) indicates that it may not exist.
        """
        tb = get_tool_box(X)
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
            if self.mode == consts.Mode_DL:
                if forecast_start is not None:
                    self.history = copy.deepcopy(forecast_start)
                    X_timestamp_start = tb.to_datetime(tb.df_to_array(X[self.timestamp])[0])
                    forecast_timestamp_end = tb.to_datetime(tb.df_to_array(forecast_start[self.timestamp])[-1])
                    if X_timestamp_start <= forecast_timestamp_end:
                        raise ValueError(f'The start date of X [{X_timestamp_start}] should be after '
                                         f'the end date of forecast_start [{forecast_timestamp_end}].')
                if self.task in consts.TASK_LIST_FORECAST:
                    self.__preprocess_forecast_start(self.history)

            if X[self.timestamp].dtypes == object:
                X[self.timestamp] = tb.to_datetime(X[self.timestamp])

            if X.shape[0] >= 3:
                X_transformed = tb.smooth_missed_ts_rows(X, ts_name=self.timestamp)
            else:
                X_transformed = X.copy()

            y_pred = self.sk_pipeline.predict(X_transformed)

            if self.task in consts.TASK_LIST_DETECTION:
                y_proba = self.sk_pipeline.predict_proba(X_transformed)
                forecast = tb.DataFrame(y_pred, columns=[consts.ANOMALY_LABEL])
                forecast[consts.ANOMALY_CONFIDENCE] = y_proba[:, 1]
            else:
                forecast = tb.DataFrame(y_pred, columns=self.target)
            forecast = tb.concat_df([X_transformed[[self.timestamp]], forecast], axis=1)
            forecast = tb.join_df(X[[self.timestamp]], forecast, on=self.timestamp)

            if self.kwargs.get('train_end_date') is not None:
                forecast = tb.drop(forecast, columns=[self.timestamp])

            return forecast
        else:
            y_pred = self.sk_pipeline.predict(X)
            return y_pred

    def predict_proba(self, X):
        """Predicts target probabilities for sequences in X for classification task.

        Parameters
        ----------
        X: 'DataFrame'.
            X.columns = [variate_1, variate_2,...].
        """
        if self.task in consts.TASK_LIST_CLASSIFICATION + consts.TASK_LIST_DETECTION:
            y_proba = self.sk_pipeline.predict_proba(X)
        else:
            raise ValueError('predict_proba is used for classification only.')
        return y_proba

    def evaluate(self, y_true, y_pred, y_proba=None, metrics=None):
        """Evaluates model performance.

        Parameters
        ----------
        y_true: 'np.arrray'.
        y_pred: 'pd.DataFrame' or 'np.arrray'.
            For forecast task, 'pd.DataFrame', X.columns could be ['timestamp', (covariate_1),
            (covariate_2),..., variate_1, variate_2,...].
            (covariate_1) indicates that it may not exist.
            For classification and regression tasks, 'np.arrray'.
        y_proba: 'np.arrray' or None, some metrics should be given, such as AUC.
        metrics: list, tuple or None. If None,
            For forecast or regression tasks, metrics = ['mae', 'mse', 'rmse', 'mape', 'smape'],
            For classification task, metrics = ['accuracy', 'f1', 'precision', 'recall'].
        """

        import pandas as pd
        from hyperts.utils.metrics import calc_score

        pd.set_option('display.max_columns', 10,
                      'display.max_rows', 10,
                      'display.float_format', lambda x: '%.4f' % x)

        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        y_proba = np.array(y_proba) if not isinstance(y_proba, np.ndarray) else y_proba

        if self.task in consts.TASK_LIST_FORECAST+consts.TASK_LIST_REGRESSION and metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape', 'smape']
        elif self.task in consts.TASK_LIST_CLASSIFICATION and metrics is None:
            metrics = ['accuracy', 'f1', 'precision', 'recall']
        elif self.task in consts.TASK_LIST_DETECTION and metrics is None:
            metrics = ['f1', 'precision', 'recall', 'roc_auc_score']

        tb = get_tool_box(y_pred)
        if self.task in consts.TASK_LIST_FORECAST:
            y_pred = tb.df_to_array(y_pred[self.target])

        if self.task in consts.TASK_LIST_DETECTION:
            y_pred = tb.df_to_array(y_pred[[consts.ANOMALY_LABEL]])

        if 'binaryclass' in self.task or self.task in consts.TASK_LIST_DETECTION:
            task = 'binary'
        elif 'multiclass' in self.task:
            task = 'multiclass'
        else:
            task = self.task

        scores = calc_score(y_true, y_pred, y_proba,
                            metrics=metrics,
                            task=task,
                            pos_label=self.kwargs.get('pos_label'))

        scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        scores = scores.reset_index().rename(columns={'index': 'Metirc'})

        return scores

    def make_future_dataframe(self, periods=None, start_date=None, covariate_df=None):
        """Simulate the trend using the extrapolated generative model.

        Notes
        ----------
        If covariate_df exists, the start time of the covariate_df must be the same as that of start_date.
        In addition, the covariate_df length is not less than periods.

        Parameters
        ----------
        periods: 'int', number of periods to forecast forward.
        start_date: 'str' or datetime-like, forecast the future start date.

        Returns
        -------
        pd.Dataframe that extends forward from the end of history for the
        requested number of periods.
        """
        tb = get_tool_box(self.history)
        if self.kwargs.get('train_end_date') is None:
            start_date = self.history[self.timestamp].max() if start_date is None else start_date
        else:
            start_date = self.kwargs.get('train_end_date')
        dates = tb.date_range(start=start_date, periods=periods+1, freq=self.freq)
        dates = dates[dates > start_date]
        dates = dates[:periods]
        futures = tb.DataFrame({self.timestamp: dates})

        if covariate_df is not None and len(covariate_df) >= len(futures):
            covariate_df = tb.reset_index(covariate_df[self.covariables])
            covariate_df = tb.select_1d_forward(covariate_df, periods)
            futures = tb.concat_df([futures, covariate_df], axis=1)

        return futures

    def plot(self,
             forecast,
             actual=None,
             history=None,
             var_id=0,
             show_forecast_interval=True,
             interactive=True,
             figsize=None):
        """Plots forecast trend curves for the forecst task.

        Notes
        ----------
        1. This function can plot the curve of only one target variable. If not specified,
        index 0 is ploted by default.

        2. This function supports ploting of historical observations, future actual values,
        and forecast intervals.

        Parameters
        ----------
        forecast: 'DataFrame'. The columns need to include the timestamp column
            and the target columns.
        actual: 'DataFrame' or None. If it is not None, the column needs to include
            the time column and the target column.
        var_id: 'int' or 'str'. If int, it is the index of the target column. If str,
            it is the name of the target column. default 0.
        show_forecast_interval: 'bool'. Whether to show the forecast intervals.
            Default True.

        Returns
        ----------
        fig : 'plotly.graph_objects.Figure'.
        """
        history = history if history is not None else self.history

        if self.task in consts.TASK_LIST_FORECAST:
            forecast_interval = self.forecast_interval(forecast)
        else:
            forecast_interval = None

        if self.task in consts.TASK_LIST_DETECTION:
            assert actual is not None, 'actual can not be None in anomaly detection task.'

        plot_options = {
            'task': self.task,
            'anomaly_detection_label': self.kwargs.get('anomaly_detection_label'),
        }

        if interactive and enable_plotly:
            plot_plotly(forecast,
                        timestamp_col=self.timestamp,
                        target_col=self.target,
                        actual=actual,
                        var_id=var_id,
                        history=history,
                        forecast_interval=forecast_interval,
                        show_forecast_interval=show_forecast_interval,
                        include_history=False if history is None else True,
                        **plot_options)
        elif not interactive and enable_mpl:
            plot_mpl(forecast,
                     timestamp_col=self.timestamp,
                     target_col=self.target,
                     actual=actual,
                     var_id=var_id,
                     history=history,
                     forecast_interval=forecast_interval,
                     show_forecast_interval=show_forecast_interval,
                     include_history=False if history is None else True,
                     figsize=figsize,
                     **plot_options)
        else:
            raise ValueError('No install matplotlib or plotly.')

    def forecast_interval(self, forecast, confidence_level=0.9):
        """infer forecast interval.

        Parameters
        ----------
        forecast: 'DataFrame'. The columns need to include the timestamp column
            and the target columns.
        confidence_level: float, default 0.9.
        """
        tb = get_tool_box(forecast)
        return tb.infer_forecast_interval(forecast[self.target], *self.prior, confidence_level=confidence_level)


    def split_X_y(self, data, smooth=False, impute=False):
        """Splits the data into X and y.

        Parameters
        ----------
        data: 'DataFrame', including X and y.
        smooth: Whether to smooth missed time series rows. Default False.
            Example:
                TimeStamp      y
                2021-03-01    3.4
                2021-03-02    5.2
                2021-03-04    6.7
                2021-03-05    2.3
                >>
                TimeStamp      y
                2021-03-01    3.4
                2021-03-02    5.2
                2021-03-03    NaN
                2021-03-04    6.7
                2021-03-05    2.3
        impute: Whether to impute in missing values. Default False.
            Example:
                TimeStamp      y
                2021-03-01    3.4
                2021-03-02    5.2
                2021-03-03    NaN
                2021-03-04    6.7
                2021-03-05    2.3
                >>
                TimeStamp      y
                2021-03-01    3.4
                2021-03-02    5.2
                2021-03-03    3.4
                2021-03-04    6.7
                2021-03-05    2.3
        Returns
        -------
        X, y.
        """
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
            tb = get_tool_box(data)
            data = tb.reset_index(data)
            is_skip = self.timestamp in tb.columns_values(data)
            if self.kwargs.get('train_end_date') is not None and not is_skip:
                pseudo_timestamp = tb.DataFrame({f'{self.timestamp}':
                                   tb.date_range(start=self.kwargs.get('train_end_date'),
                                                 periods=len(data)+1,
                                                 freq=self.kwargs.get('generate_freq'))})
                pseudo_timestamp = tb.select_1d_reverse(pseudo_timestamp, len(data))
                pseudo_timestamp = tb.reset_index(pseudo_timestamp)
                data = tb.concat_df([pseudo_timestamp, data], axis=1)

            if self.covariables is not None:
                excluded_variables = tb.list_diff(tb.columns_values(data), self.target)
            else:
                excluded_variables = [self.timestamp]

            data = tb.sort_values(data, ts_name=self.timestamp)

            data = tb.drop_duplicated_ts_rows(data, ts_name=self.timestamp)

            if smooth is not False:
                data = tb.smooth_missed_ts_rows(data, ts_name=self.timestamp, freq=self.freq)

            if impute is not False:
                data[self.target] = tb.multi_period_loop_imputer(data[self.target], freq=self.freq)

            if self.task in consts.TASK_LIST_FORECAST:
                X, y = data[excluded_variables], data[self.target]
            else:
                X = data
                if self.kwargs.get('anomaly_label_col') is not None:
                    y = X.pop(self.kwargs.get('anomaly_label_col'))
                else:
                    y = None
        else:
            X = data
            y = X.pop(self.target)
        return X, y

    def get_pipeline_params(self):
        """Gets sklearn pipeline parameters.

        """
        return self.sk_pipeline.get_params

    def __preprocess_forecast_start(self, forecast_start):
        """Performs data preprocessing for the external forecast_start.

        Parameters
        ----------
        forecast_start : 'DataFrame'. This parameter applies only to 'dl' mode.
            Forecast the start fragment, if None, by default the last window fragment of the
            train data.
            forecast_start.columns = ['timestamp', (covariate_1), (covariate_2),...].
            (covariate_1) indicates that it may not exist.
        """

        def preprocess(X, y, estimator):
            X = copy.deepcopy(X)
            y = copy.deepcopy(y)
            X = estimator.data_pipeline.transform(X)
            y = estimator.model.transform(y)
            X, y = estimator.model.model.meta.transform(X, y)
            forecast_start = tb.concat_df([X, y], axis=1)

            window_length = estimator.model.model.window
            cont_column_names = estimator.model.model.meta.cont_column_names
            cat_column_names = estimator.model.model.meta.cat_column_names
            continuous_length = len(cont_column_names)
            categorical_length = len(cat_column_names)
            column_names = cont_column_names + cat_column_names
            data = forecast_start.drop([self.timestamp], axis=1)
            data = tb.df_to_array(data[column_names]).astype(consts.DATATYPE_TENSOR_FLOAT)
            forecast_start = data[-window_length:].reshape(1, window_length, data.shape[1])
            if categorical_length != 0:
                X_cont_start = forecast_start[:, :, :continuous_length]
                X_cat_start = forecast_start[:, :, continuous_length:]
                forecast_start = [X_cont_start, X_cat_start]

            return forecast_start

        # 1. transform
        tb = get_tool_box(forecast_start)
        forecast_start = self.sk_pipeline.named_steps.data_preprocessing.transform(forecast_start)
        X, y = self.split_X_y(forecast_start, smooth=True, impute=True)

        # 2. perprocessing
        if hasattr(self.sk_pipeline.named_steps.estimator, 'ensemble_size'):
            for estimator in self.sk_pipeline.named_steps.estimator.estimators:
                if estimator is not None:
                    forecast_start = preprocess(X, y, estimator)
                    estimator.model.model.forecast_start = forecast_start
        else:
            forecast_start = preprocess(X, y, self.sk_pipeline.named_steps.estimator)
            self.sk_pipeline.named_steps.estimator.model.model.forecast_start = forecast_start

    def save(self, model_file):
        """Save TSPipeline.

        Parameters
        ----------
        Parameters
        ----------
        model_file: str, the path where the file is saved. For example, "home/xxx/xxx/models".
        """
        from hyperts.utils.models import save_model
        save_model(model=self, model_file=model_file)

    @staticmethod
    def load(model_file):
        """Load TSPipeline.

        Parameters
        ----------
        model_file: str, the path where the file is saved. For example,
            1. "home/xxx/xxx/models/stats_models"
            2. "home/xxx/xxx/models/dl_models"
            3. "home/xxx/xxx/models/nas_models"
        """
        from hyperts.utils.models import load_model
        load_model(model_file)


class TSCompeteExperiment(SteppedExperiment):
    """A powerful experiment strategy for Automatic Time Series with a set of advanced features.

    Parameters
    ----------
    hyper_model: hypernets.model.HyperModel
        A `HyperModel` instance
    X_train: Pandas or Dask DataFrame
        Feature data for training
    y_train: Pandas or Dask Series
        Target values for training
    X_eval: (Pandas or Dask DataFrame) or None
        (default=None), Feature data for evaluation
    y_eval: (Pandas or Dask Series) or None, default None.
        Target values for evaluation
    X_test: (Pandas or Dask Series) or None, default None.
        Unseen data without target values for semi-supervised learning
    eval_size: 'float' or 'int', default None.
        Only valid when ``X_eval`` or ``y_eval`` is None. If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the eval split. If int, represents the absolute number of
        test samples. If None, the value is set to the complement of the train size.
    freq: 'str', DateOffset or None, default None.
    target_col: 'str' or list[str], default None.
    timestamp_col: str or None, default None.
    covariate_cols: list[list or None, list or None] or None, default None. covariate_cols needs to contain
        a list of original covariates and a list of cleaned covariates.
    covariate_data_cleaner_args: 'dict' or None, default None. Suitable for forecast task.
        Dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized
        with default values.
    data_cleaner_args: 'dict' or None, default None. Suitable for classification/regression tasks.
        Dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized
        with default values.
    cv: 'bool', default False.
        If True, use cross-validation instead of evaluation set reward to guide the search process.
    num_folds: 'int', default 3.
        Number of cross-validated folds, only valid when cv is true.
    task: 'str' or None, default None.
        Task could be 'univariate-forecast', 'multivariate-forecast', and 'univariate-binaryclass', etc.
        See consts.py for details.
    mode : str, default 'stats'. Optional {'stats', 'dl', 'nas'}, where,
        'stats' indicates that all the models selected in the execution experiment are statistical models.
        'dl' indicates that all the models selected in the execution experiment are deep learning models.
        'nas' indicates that the selected model of the execution experiment will be a deep network model
        for neural architecture search, which is not currently supported.
    id: trial id, default None.
    callbacks: list of callback functions or None, default None.
        List of callback functions that are applied at each experiment step. See `hypernets.experiment.ExperimentCallback`
        for more information.
    log_level: 'int', 'str', or None, default None,
        Level of logging, possible values:
            -logging.CRITICAL
            -logging.FATAL
            -logging.ERROR
            -logging.WARNING
            -logging.WARN
            -logging.INFO
            -logging.DEBUG
            -logging.NOTSET
    random_state: 'int' or RandomState instance, default None.
        Controls the shuffling applied to the data before applying the split.
    scorer: 'str', callable or None, default None.
        Scorer to used for feature importance evaluation and ensemble. It can be a single string
        (see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html))
        or a callable (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)).
        Will be inferred from *hyper_model.reward_metric* if it's None.
    ensemble_size: 'int', default 10.
        The number of estimator to ensemble. During the AutoML process, a lot of models will be generated with different
        preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models
        that perform well to ensemble can obtain better generalization ability than just selecting the single best model.
    final_retrain_on_wholedata: bool, after the search, whether to retrain the optimal model on the whole data set.
        default True.
    """

    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None,
                 eval_size=consts.DEFAULT_EVAL_SIZE,
                 freq=None,
                 target_col=None,
                 timestamp_col=None,
                 covariate_cols=None,
                 anomaly_label_col=None,
                 covariate_cleaner=None,
                 cv=False,
                 num_folds=3,
                 task=None,
                 mode='stats',
                 id=None,
                 forecast_train_data_periods=None,
                 hist_store_upper_limit=consts.HISTORY_UPPER_LIMIT,
                 contamination=consts.CONTAMINATION,
                 callbacks=None,
                 log_level=None,
                 random_state=None,
                 scorer=None,
                 optimize_direction=None,
                 ensemble_size=10,
                 final_retrain_on_wholedata=True,
                 **kwargs):

        if random_state is None:
            random_state = np.random.randint(0, 65535)
        set_random_state(random_state)

        if task is None:
            task = hyper_model.task

        self.cv = cv
        self.freq = freq
        self.task = task
        self.mode = mode
        self.target = target_col
        self.timestamp = timestamp_col
        self.max_window_length = hist_store_upper_limit
        self.train_end_date = kwargs.pop('train_end_date', None)
        self.generate_freq = kwargs.pop('generate_freq', None)
        self.pos_label = kwargs.get('pos_label', None)
        if self.task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
            tb = get_tool_box(X_train, y_train)
            all_data = tb.concat_df([X_train, y_train], axis=1)
            if X_eval is not None or y_eval is not None:
                eval_data = tb.concat_df([X_eval, y_eval], axis=1)
                all_data = tb.concat_df([all_data, eval_data], axis=0)
            self.history = tb.select_1d_reverse(all_data, hist_store_upper_limit)
        else:
            self.history = None

        if covariate_cols is not None and len(covariate_cols) == 2:
            self.covariables = covariate_cols[0]
            cleaned_covariables = covariate_cols[1]
        elif covariate_cols is not None and len(covariate_cols) != 2:
            raise ValueError('covariate_cols needs to contain a list of original '
                             'covariates and a list of cleaned covariates.')
        else:
            self.covariables = None
            cleaned_covariables = None

        self.anomaly_label_col = anomaly_label_col

        steps = []

        # data clean and preprocessing
        if task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_DETECTION:
            steps.append(TSAFDataPreprocessStep(self, consts.StepName_DATA_PREPROCESSING,
                                               freq=freq,
                                               cv=cv,
                                               ensemble_size=ensemble_size,
                                               timestamp_col=timestamp_col,
                                               covariate_cols=cleaned_covariables,
                                               covariate_cleaner=covariate_cleaner,
                                               anomaly_label_col=anomaly_label_col,
                                               contamination=contamination,
                                               train_data_periods=forecast_train_data_periods))
        else:
            steps.append(TSRCDataPreprocessStep(self, consts.StepName_DATA_PREPROCESSING,
                                               cv=cv))

        # model selection and hyper-parameters search step
        steps.append(TSSpaceSearchStep(self, consts.StepName_SPACE_SEARCHING,
                                       cv=cv,
                                       num_folds=num_folds))

        if ensemble_size is not None and ensemble_size > 1:
            # model ensemble step
            tb = get_tool_box(X_train, y_train)
            if scorer is None:
                kwargs['pos_label'] = tb.infer_pos_label(y_train, task, kwargs.get('pos_label'))
                scorer = tb.metrics.metric_to_scorer(hyper_model.reward_metric, task=task,
                 pos_label=kwargs.get('pos_label'), optimize_direction=optimize_direction)
            steps.append(TSEnsembleStep(self, consts.StepName_FINAL_ENSEMBLE,
                                        mode=mode,
                                        scorer=scorer,
                                        ensemble_size=ensemble_size,
                                        cv=cv,
                                        retrain_on_wholedata=final_retrain_on_wholedata))
        else:
            # final train step
            steps.append(TSFinalTrainStep(self, consts.StepName_FINAL_TRAINING,
                                          mode=mode,
                                          retrain_on_wholedata=final_retrain_on_wholedata))

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        if log_level is not None:
            _set_log_level(log_level)

        self.run_kwargs = kwargs
        super(TSCompeteExperiment, self).__init__(steps,
                                           hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                           X_test=X_test, eval_size=eval_size, task=task,
                                           id=id,
                                           callbacks=callbacks,
                                           random_state=random_state)

    def run(self, **kwargs):
        """Related parameters that can be reset, mainly involving pipeline stap and model parameters.
           Including but not limited to the following:

        Parameters
        ----------
        max_trials: int, maximum number of tests (model search), optional, (default=3).
        final_train_epochs: int, the searched best DL model performs the final training epochs, optional, (default=120).
        epochs: int, DL model training epochs in each trail, optional, (default=120).
        batch_size: int, number of samples per gradient update, optional, (default is self-adaption).
        optimizer: str, the optimizer of DL model, optional {'adam', 'sgd', 'rmsprop'}, (default='adam').
        learning_rate: float, the optimizer's learning rate, optional, (default=0.001).
        loss: str, the loss function of DL model, optional, (default is searching).
        drop_rate: float, The rate of Dropout for DL model, optional, (default is searching).
        dl_forecast_window : int, the sequence length of each sample in DL model, i.e., timestamp of
            [batch_size, timestamp, dim], optional, (default is searching).
        dl_forecast_length: int, the forecast length for the future in DL model, optional, (default is searching).
        seasonality_mode: str, 'additive' (default) or 'multiplicative' in Prophet.
        period_offset: int, s of seasonal_order(P,D,Q,s) in ARIMA, optional, (default is searching).
        maxlags: int, maximum number of lags to check for order selection in VAR, optional, (default is searching).
        ...

        Notes
        ----------
        Once the above parameters are specified, the search space cannot be searched.
        """
        kwargs = self._reset_run_kwargs(**kwargs)
        run_kwargs = {**self.run_kwargs, **kwargs}
        return super().run(**run_kwargs)

    def to_estimator(self, X_train, y_train, X_test, X_eval, y_eval, steps):
        sk_pipeline = super(TSCompeteExperiment, self).to_estimator(
                            X_train, y_train, X_test, X_eval, y_eval, steps)

        return TSPipeline(sk_pipeline,
                          freq=self.freq,
                          task=self.task,
                          mode=self.mode,
                          timestamp=self.timestamp,
                          covariables=self.covariables,
                          target=self.target,
                          history=self.history,
                          train_end_date=self.train_end_date,
                          generate_freq=self.generate_freq,
                          pos_label=self.pos_label,
                          anomaly_label_col=self.anomaly_label_col)

    def report_best_trial_params(self):
        """Gets experiment best trial parameters.

        """
        if self.hyper_model_ is not None:
            best_trial = self.hyper_model_.get_best_trial()
            best_trial_params = best_trial.to_df(include_params=True).T
            best_trial_params.rename(columns={0: 'value'}, inplace=True)
            return best_trial_params
        else:
            raise RuntimeError('The best trial parameters can be obtained only after running.')

    def _repr_html_(self):
        return self.__repr__()

    def _reset_run_kwargs(self, **kwargs):
        if kwargs.get('dl_forecast_window') is not None:
            if kwargs.get('dl_forecast_window') <= self.max_window_length:
                kwargs['window'] = kwargs.pop('dl_forecast_window')
            else:
                raise ValueError(f'dl_window_length cannot be greater than {self.max_window_length}.')
        if kwargs.get('dl_forecast_length', 0) > 0:
            kwargs['forecast_length'] = kwargs.pop('dl_forecast_length')

        if self.cv and kwargs.get('epochs') is None:
            kwargs['epochs'] = consts.TRAINING_EPOCHS // 2

        return kwargs