# -*- coding:utf-8 -*-
import pandas as pd
from os.path import join, dirname
from sklearn.metrics.pairwise import cosine_similarity

from hyperts.utils import consts, get_tool_box

from hyperts.framework.meta_learning import normalization, metric_mapping_dict
from hyperts.framework.meta_learning.tsfeatures import metafeatures_from_timeseries

from hypernets.utils import logging

logger = logging.get_logger(__name__)

class TrialInstance:
    """
    Trial Instance.

    Parameters
    ----------
    signature: str, trial instance signature.
    vectors: list, configuration vectors of space_sample.
    reward: float, reward of space_sample.
    elapsed: float, trial elapsed.
    run: bool, whether the current data has been run (fit), default False.
    """
    def __init__(self, signature, vectors, reward, elapsed, run=False):
        self.signature = signature
        self.vectors = vectors
        self.reward = reward
        self.elapsed = elapsed
        self.run = run

    def __repr__(self):
        return f"signature: {self.signature}\n" \
               f"vectors: {self.vectors}, reward: {self.reward}, elapsed: {self.elapsed}."


class TrialStore:
    """
    Trial store for meta learning to warm start optimization.

    Parameters
    ----------
    task: str, task name, for example, 'univariate-forecast', 'multivariate-forecast' and so on.
    dataset_id: str, dataset id based on shape and dtype (X, y).
    reward_metric: 'str' or callable, default 'accuracy' for binary/multiclass task, 'rmse' for
        forecast/regression task.
    is_scale: bool, whether to scale metafeatures, default True.
    trials_limit: int, number of collection trials, default 30.
    """
    def __init__(self, task, dataset_id, reward_metric, is_scale=True, trials_limit=30, **kwargs):
        self.task = task
        self.dataset_id = dataset_id
        self.reward_metric = reward_metric
        self.is_scale = is_scale
        self.trials_limit = trials_limit
        self.timestamp = kwargs.get('timestamp')

        self.trials = []
        self.datasetnames = None

    def fit(self, X, y=None):
        """
        Calculate metafeatures and configuration information.
        """
        tb = get_tool_box()
        if y is not None:
            metadata = tb.concat_df([X, y], axis=1)
        else:
            metadata = X.copy()

        if self.dataset_id is not None:
            self.dataset_id = tb.data_hasher()([X, y])

        metafeatures = self.get_metafeatures()
        if 'dataset_name' in metafeatures.columns.to_list():
            self.datasetnames = metafeatures.pop('dataset_name')

        metafeature = metafeatures_from_timeseries(metadata, self.timestamp, scale_ts=True)
        metafeature.rename(index={0: self.dataset_id}, inplace=True)

        if metafeature.index[0] not in metafeatures.index.to_list():
            metafeatures = tb.concat_df([metafeature, metafeatures], axis=0)
        else:
            metafeatures.drop(index=metafeature.index[0], inplace=True)
            metafeatures = tb.concat_df([metafeature, metafeatures], axis=0)

        if self.is_scale:
            metafeatures = normalization(metafeatures)

        metafeatures = metafeatures.dropna(axis=1, how='all')

        similarity = cosine_similarity(metafeatures)
        mf_index = metafeatures.index.to_list()
        similarity = tb.DataFrame(similarity, columns=mf_index, index=mf_index)

        similarity_sorted = similarity.sort_values(metafeature.index, ascending=False)
        similarity_sorted = similarity_sorted.loc[:, metafeature.index]

        configurations = self.get_configurations(self.reward_metric)

        for cidx, sim_idx in enumerate(similarity_sorted.index.to_list()):
            if len(self.trials) < self.trials_limit:
                subcfgs = configurations[configurations['dataset_id'] == sim_idx]
                if subcfgs.shape[0] >= 1:
                    for i in range(subcfgs.shape[0]):
                        signature = str(subcfgs.iloc[i]['signature'])
                        vectors = list(map(int, subcfgs.iloc[i]['vectors'][1:-1].split(',')))
                        reward = float(subcfgs.iloc[i]['reward'])
                        elapsed = float(subcfgs.iloc[i]['elapsed'])
                        trial = TrialInstance(signature, vectors, reward, elapsed)
                        self.trials.append(trial)

        logger.info(f'{len(self.trials)} similar trials were collected.')

        return self

    def get(self, dataset_id, space_sample):
        """
        Extract meta trial from trial store.
        """
        assert self.dataset_id == dataset_id
        for trial in self.trials:
            if trial.run and trial.signature == space_sample.signature and \
                    trial.vectors == space_sample.vectors:
                return trial
        return None

    def put(self, dataset_id, new_trial):
        """
        Put trial into trials.
        """
        assert self.dataset_id == dataset_id

        signature = new_trial.space_sample.signature
        vectors = new_trial.vectors
        reward = new_trial.reward
        elapsed = new_trial.elapsed

        i = 0
        for t in self.trials:
            if i < self.trials_limit and t.signature == signature and t.vectors == vectors:
                self.trials[i].reward = reward
                self.trials[i].elapsed = elapsed
                self.trials[i].run = True
                break
            i += 1

        if i == self.trials_limit:
            trial = TrialInstance(signature, vectors, reward, elapsed)
            self.trials.append(trial)

    def get_all(self, dataset_id, space_signature):
        """
        Extract all trial configurations that satisfy the specified space_sample.
        """
        assert self.dataset_id == dataset_id
        all_suggested_trials = []

        for trial in self.trials:
            if trial.signature == space_signature:
                all_suggested_trials.append((trial.vectors, trial.reward))

        return all_suggested_trials

    def get_metafeatures(self, features=None):
        """
        Extract meta-features.
        """
        module_path = dirname(__file__)
        meta_path_file = join(module_path, 'metafeatures')
        if self.task == consts.Task_UNIVARIATE_FORECAST:
            meta_path_file = join(meta_path_file, 'metafeatures_univariate_forecast.csv')
        elif self.task == consts.Task_MULTIVARIATE_FORECAST:
            meta_path_file = join(meta_path_file, 'metafeatures_multivariate_forecast.csv')
        else:
            raise RuntimeError(f'No support task: {self.task}.')

        metafeatures = pd.read_csv(meta_path_file, index_col=0)

        if features is not None:
            metafeatures = metafeatures.loc[:, features]

        logger.info('Extract meta features finished.')

        return metafeatures

    def get_configurations(self, reward_metric):
        """
        Extract the trial configurations of history.
        """
        module_path = dirname(__file__)
        meta_path_file = join(module_path, 'configurations')
        if self.task == consts.Task_UNIVARIATE_FORECAST:
            meta_path_file = join(meta_path_file, 'configurations_univariate_forecast.csv')
        elif self.task == consts.Task_MULTIVARIATE_FORECAST:
            meta_path_file = join(meta_path_file, 'configurations_multivariate_forecast.csv')
        else:
            raise RuntimeError(f'No support task: {self.task}.')

        configurations = pd.read_csv(meta_path_file, index_col=0)

        if isinstance(reward_metric, str):
            reward_metric = metric_mapping_dict.get(reward_metric)
        elif callable(reward_metric):
            reward_metric = metric_mapping_dict.get(reward_metric.__name__)
        else:
            raise ValueError('The reward_metric definition might be wrong.')

        if reward_metric is not None:
            configurations = configurations[configurations.reward_metric == reward_metric]

        if configurations.empty:
            raise ValueError(f'Failed to extract the configuration matching {reward_metric}.')

        logger.info('Extract trial configurations finished.')

        return configurations