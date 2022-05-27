# -*- coding:utf-8 -*-
import pandas as pd
from os.path import join, dirname
from sklearn.metrics.pairwise import cosine_similarity

from hyperts.utils import consts, get_tool_box
from .tsfeatures import metafeatures_from_timeseries

from hypernets.utils import logging

logger = logging.get_logger(__name__)

class TrialInstance:
    """
    Trial Instance.

    Parameters
    ----------
    vectors: list, configuration vectors of space_sample.
    reward: float, reward of space_sample.
    """
    def __init__(self, signature, vectors, reward):
        self.signature = signature
        self.vectors = vectors
        self.reward = reward

    def __repr__(self):
        return f"signature: {self.signature}\n" \
               f"vectors: {self.vectors}, reward: {self.reward}"


class TrialStore:
    """
    Trial store for meta learning to warm start optimization.

    Parameters
    ----------
    dataset_id: str, dataset id based on shape and dtype (X, y).
    """
    def __init__(self, task, dataset_id, **kwargs):
        self.task = task
        self.dataset_id = dataset_id
        self.timestamp = kwargs.get('timestamp')

        self.trials = []

    def fit(self, X, y=None):
        """
        Calculate metafeatures and configuration information.
        """
        tb = get_tool_box()
        if y is not None:
            metadata = tb.concat_df([X, y], axis=1)
        else:
            metadata = X.copy()

        metafeatures = self.get_metafeatures()
        metafeature = metafeatures_from_timeseries(metadata, self.timestamp)

        if metafeature.index  not in metafeatures.index:
            metafeatures = tb.concat_df([metafeature, metafeatures], axis=0)
        else:
            metafeatures.drop(0, inplace=True)
            metafeatures = tb.concat_df([metafeature, metafeatures], axis=0)

        similarity = cosine_similarity(metafeatures)

        similarity_sorted = similarity.sort_values(metafeature.index, ascending=False)
        similarity_sorted = similarity_sorted.loc[:, metafeature.index]

        configurations = self.get_configurations()

        for sim_idx in similarity_sorted.index.to_list():
            configuration = configurations.loc[sim_idx]
            trial = TrialInstance(signature=configuration['signature'],
                                  vectors=configuration['signature'],
                                  reward=configuration['reward'])
            self.trials.append(trial)

        return self


    def get_all(self, dataset_id, space_signature):
        """
        Extract all trial configurations that satisfy the specified space_sample.
        """
        assert self.dataset_id == dataset_id

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
            raise RuntimeError(f'No support {self.task}.')

        metafeatures = pd.read_csv(meta_path_file)

        if features is not None:
            metafeatures = metafeatures.loc[:, features]

        return metafeatures

    def get_configurations(self):
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
            raise RuntimeError(f'No support {self.task}.')

        configurations = pd.read_csv(meta_path_file)

        return configurations