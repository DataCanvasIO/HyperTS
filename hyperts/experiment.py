import copy

import numpy as np
import pandas as pd

from hypernets.core import set_random_state
from hypernets.experiment.compete import SteppedExperiment, ExperimentStep, \
                                         EnsembleStep, FinalTrainStep

from hypernets.utils import logging
from hypernets.tabular import get_tool_box
from hypernets.tabular.data_cleaner import DataCleaner

from hyperts.utils import data_ops as dp, consts

logger = logging.get_logger(__name__)

DEFAULT_EVAL_SIZE = 0.2

def _set_log_level(log_level):
    logging.set_level(log_level)


class TSDataPreprocessStep(ExperimentStep):
    def __init__(self, experiment, name, timestamp_col=None, freq=None,
                 covariate_cols=None, covariate_data_clean_args=None):
        super().__init__(experiment, name)

        timestamp_col = [timestamp_col] if isinstance(timestamp_col, str) else timestamp_col
        covariate_cols = [covariate_cols] if isinstance(covariate_cols, str) else covariate_cols

        self.freq = freq
        self.timestamp_col = timestamp_col if timestamp_col is not None else consts.TIMESTAMP
        self.covariate_cols = covariate_cols
        self.covariate_data_clean_args = covariate_data_clean_args if covariate_data_clean_args is not None else {}
        self.covariate_data_clean_args.update({'correct_object_dtype': False})
        # fitted
        self.covariate_data_cleaner = DataCleaner(**self.covariate_data_clean_args)

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)
        # 1. covariate variables data clean procsss
        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            X_train = self.covariate_transform(X_train, training=True)

        # 2. target plus covariable process
        train_Xy = pd.concat([X_train, y_train], axis=1)
        variable_cols = dp.list_diff(train_Xy.columns, self.timestamp_col)
        target_variable_cols = dp.list_diff(variable_cols, self.covariate_cols)
        excluded_cols = dp.list_diff(train_Xy.columns, target_variable_cols)
        train_Xy = self.series_transform(train_Xy, target_variable_cols)
        X_train, y_train = train_Xy[excluded_cols], train_Xy[target_variable_cols]

        # 3. eval variables data process
        if X_eval is None or y_eval is None:
            eval_size = self.experiment.eval_size
            if self.task in [consts.TASK_FORECAST, consts.TASK_UNIVARIABLE_FORECAST, consts.TASK_MULTIVARIABLE_FORECAST]:
                X_train, X_eval, y_train, y_eval = \
                    dp.temporal_train_test_split(X_train, y_train, test_size=eval_size)
        else:
            if self.covariate_cols is not None and len(self.covariate_cols) > 0:
                X_eval = self.covariate_transform(X_eval, training=False)
            eval_Xy = pd.concat([X_eval, y_eval], axis=1)
            eval_Xy = self.series_transform(eval_Xy, target_variable_cols)
            X_eval, y_eval = eval_Xy[excluded_cols], eval_Xy[target_variable_cols]

        # 4. compute new data shape
        data_shapes = {'X_train.shape': X_train.shape,
                       'y_train.shape': y_train.shape,
                       'X_eval.shape': None if X_eval is None else X_eval.shape,
                       'y_eval.shape': None if y_eval is None else y_eval.shape,
                       'X_test.shape': None if X_test is None else X_test.shape
                       }

        # 5. reset part parameters
        self.data_shapes = data_shapes

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            X_transform = self.covariate_transform(X, training=False)
            X_transform = self.series_transform(X_transform)
        else:
            X_transform = self.series_transform(X)
        return X_transform

    def covariate_transform(self, X, training=False):
        df_timestamp = X[self.timestamp_col]
        if training:
            df_covariate, _ = self.covariate_data_cleaner.fit_transform(X[self.covariate_cols])
        else:
            df_covariate = self.covariate_data_cleaner.transform(X[self.covariate_cols])
        assert df_covariate.shape[0] == X.shape[0], \
            'The row of clearned covariable is not equal the row of X_train.'
        X = pd.concat([df_timestamp, df_covariate], axis=1)
        return X

    def series_transform(self, X, target_variable_cols=None):
        covar_object_names, covar_float_names = [], []

        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            for col in self.covariate_cols:
                if X[col].dtypes == consts.DATATYPE_OBJECT:
                    covar_object_names.append(col)
                elif X[col].dtypes == consts.DATATYPE_FLOAT:
                    covar_float_names.append(col)

        if target_variable_cols is not None:
            impute_col_names = target_variable_cols + covar_float_names
        else:
            impute_col_names = covar_float_names

        self.freq = self.freq if self.freq is not None else \
            dp.infer_ts_freq(X[self.timestamp_col], ts_name=self.timestamp_col[0])
        X = dp.drop_duplicated_ts_rows(X, ts_name=self.timestamp_col[0])
        X = dp.smooth_missed_ts_rows(X, freq=self.freq, ts_name=self.timestamp_col[0])

        if target_variable_cols is not None and len(target_variable_cols) > 0:
            X[target_variable_cols] = dp.nan_to_outliers(X[target_variable_cols])
        if impute_col_names is not None and len(impute_col_names) > 0:
            X[impute_col_names] = dp.multi_period_loop_imputer(X[impute_col_names], freq=self.freq)
        if covar_object_names is not None and len(covar_object_names) > 0:
            X[covar_object_names] = X[covar_object_names].fillna(method='ffill').fillna(method='bfill')

        return X

    def get_params(self, deep=True):
        params = super(TSDataPreprocessStep, self).get_params()
        params['covariate_data_clean_args'] = self.covariate_data_cleaner.get_params()
        return params

    def get_fitted_params(self):
        freq = self.freq if self.freq is not None else None
        data_shapes = self.data_shapes if self.data_shapes is not None else {}
        return {**super(TSDataPreprocessStep, self).get_fitted_params(),
                **data_shapes,
                'freq': freq}


class TSSpaceSearchStep(ExperimentStep):
    def __init__(self, experiment, name):
        super().__init__(experiment, name)
        # fitted
        self.dataset_id = None
        self.model = None
        self.history_ = None
        self.best_reward_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)
        if X_eval is not None:
            kwargs['eval_set'] = (X_eval, y_eval)
        model = copy.deepcopy(self.experiment.hyper_model)  # copy from original hyper_model instance
        model.search(X_train, y_train, X_eval, y_eval, **kwargs)

        if model.get_best_trial() is None or model.get_best_trial().reward == 0:
            raise RuntimeError('Not found available trial, change experiment settings and try again pls.')

        self.dataset_id = 'abc'  # fixme
        self.model = model
        self.history_ = model.history
        self.best_reward_ = model.get_best_trial().reward

        logger.info(f'{self.name} best_reward: {self.best_reward_}')

        return self.model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'best_reward': self.best_reward_,
                'history': self.history_,
                }


class TSEnsembleStep(EnsembleStep):

    def get_ensemble(self, estimators, X_train, y_train):
        # return GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)
        tb = get_tool_box(X_train, y_train)
        if self.task in ['forecast', "multivariate-forecast"]:
            ensemble_task = 'regression'
        else:
            ensemble_task = self.task
        return tb.greedy_ensemble(ensemble_task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)


class TSExperiment(SteppedExperiment):

    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None,
                 eval_size=0.2,
                 freq=None,
                 timestamp_col=None,
                 covariate_cols=None,
                 covariate_data_clean_args=None,
                 cv=True, num_folds=3,
                 task=None,
                 callbacks=None,
                 log_level=None,
                 random_state=None,
                 ensemble_size=3,
                 **kwargs):
        """
        Parameters

        ----------
        Return

        """
        if random_state is None:
            random_state = np.random.randint(0, 65535)
        set_random_state(random_state)

        task = hyper_model.task

        # todo: check task

        # todo: check scorer

        steps = []

        # data clean
        if task in [consts.TASK_FORECAST, consts.TASK_UNIVARIABLE_FORECAST, consts.TASK_MULTIVARIABLE_FORECAST]:
            steps.append(TSDataPreprocessStep(self, consts.StepName_DATA_PREPROCESSING,
                                             freq=freq,
                                             timestamp_col=timestamp_col,
                                             covariate_cols=covariate_cols,
                                             covariate_data_clean_args=covariate_data_clean_args))

        # search step
        steps.append(TSSpaceSearchStep(self, consts.StepName_SPACE_SEARCHING))

        # ensemble step,
        # steps.append(TSEnsembleStep(self, StepNames.FINAL_ENSEMBLE, scorer=scorer, ensemble_size=ensemble_size))

        steps.append(FinalTrainStep(self, consts.StepName_FINAL_TRAINING, retrain_on_wholedata=False))

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        if log_level is not None:
            _set_log_level(log_level)

        self.run_kwargs = kwargs
        super(TSExperiment, self).__init__(steps,
                                           hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                           X_test=X_test, eval_size=eval_size, task=task,
                                           id=id,
                                           callbacks=callbacks,
                                           random_state=random_state)

    def run(self, **kwargs):
        run_kwargs = {**self.run_kwargs, **kwargs}
        return super().run(**run_kwargs)

    def _repr_html_(self):
        try:
            from hypernets.hn_widget.hn_widget.widget import ExperimentSummary
            from IPython.display import display
            display(ExperimentSummary(self))
        except:
            return self.__repr__()
