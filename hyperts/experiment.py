import copy

import numpy as np
import pandas as pd

from hypernets.core import set_random_state
from hypernets.experiment import StepNames
from hypernets.experiment.compete import SteppedExperiment, ExperimentStep, EnsembleStep, FinalTrainStep
from hypernets.tabular import get_tool_box
from hypernets.tabular.data_cleaner import DataCleaner
from hypernets.utils import logging
from hyperts.hyper_ts import HyperTS

logger = logging.get_logger(__name__)

DEFAULT_EVAL_SIZE = 0.3


class TSDataPreprocessStep(ExperimentStep):
    def __init__(self, experiment, name, covariate_cols=None, covariate_data_clean_args=None):
        super().__init__(experiment, name)

        if covariate_data_clean_args is None:
            covariate_data_clean_args = {}

        self.covariate_cols = covariate_cols
        self.covariate_data_clean_args = covariate_data_clean_args

        # fitted
        self.covariate_data_cleaner = DataCleaner(**self.covariate_data_clean_args)

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)
        # 1. process covariate features
        if self.covariate_cols is not None and len(self.covariate_cols) > 0:
            excluded_cols = list(set(X_train.columns.tolist()) - set(self.covariate_cols))
            df_exclude = X_train[excluded_cols]
            df_covariate = self.covariate_data_cleaner.fit_transform(X_train[self.covariate_cols])
            # TODO: check shape
            X_train_cleaned_covariate = pd.concat([df_exclude, df_covariate])
            X_train = X_train_cleaned_covariate

        # 2. target plus covariable features process

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_params(self, deep=True):
        return {}

    def transform(self, X, y=None, **kwargs):
        # transform covariate features
        X_transform = self.covariate_data_cleaner.fit_transform(X)
        return X_transform[0]  # selected X

    def get_fitted_params(self):
        return {}  # TODO:


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

    def __init__(self, hyper_model, X_train, y_train, timestamp_col=None, covariate_cols=None,
                 covariate_data_clean_args=None, X_eval=None, y_eval=None, log_level=None,
                 random_state=None, ensemble_size=3, **kwargs):

        if random_state is None:
            random_state = np.random.randint(0, 65535)
        set_random_state(random_state)

        task = hyper_model.task

        # todo: check task
        # todo: check scorer

        steps = []

        # data clean
        # Fix: `df.nunique(dropna=True)` in _get_df_uniques cause
        # `TypeError: unhashable type: 'Series'` in case of nest pd.Series
        if task not in [HyperTS.TASK_BINARY_CLASSIFICATION]:
            steps.append(TSDataPreprocessStep(self, StepNames.DATA_CLEAN,
                                              covariate_data_clean_args=covariate_data_clean_args))

        # search step
        steps.append(TSSpaceSearchStep(self, StepNames.SPACE_SEARCHING))

        # ensemble step,
        # steps.append(TSEnsembleStep(self, StepNames.FINAL_ENSEMBLE, scorer=scorer, ensemble_size=ensemble_size))

        steps.append(FinalTrainStep(self, StepNames.FINAL_TRAINING, retrain_on_wholedata=False))

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        # if log_level is not None:
        #     _set_log_level(log_level)

        self.run_kwargs = kwargs
        super(TSExperiment, self).__init__(steps, hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                           eval_size=0.3, task=task, id=id, random_state=random_state)

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
