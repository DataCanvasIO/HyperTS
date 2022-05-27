# -*- coding:utf-8 -*-

from lightgbm import LGBMRegressor
from hypernets.utils import logging
from hypernets.core.meta_learner import MetaLearner as BaseMetaLearner

logger = logging.get_logger(__name__)


class MetaLearner(BaseMetaLearner):

    def __init__(self, history, dataset_id, trial_store, **kwargs):
        super(MetaLearner, self).__init__(history, dataset_id, trial_store)

    def fit(self, space_signature):

        features = self.extract_features_and_labels(space_signature)
        x = []
        y = []
        for feature, label in features:
            if label != 0:
                x.append(feature)
                y.append(label)

        store_history = self.store_history.get(space_signature)

        if self.trial_store is not None and store_history is None:
            trials = self.trial_store.get_all(self.dataset_id, space_signature)
            store_x = []
            store_y = []
            for t in trials:
                store_x.append(t.vectors)
                store_y.append(t.reward)
            store_history = (store_x, store_y)
            self.store_history[space_signature] = store_history

        if store_history is None:
            store_history = ([], [])

        store_x, store_y = store_history
        x = x + store_x
        y = y + store_y
        if len(x) >= 2:
            regressor = LGBMRegressor()
            regressor.fit(x, y)
            self.regressors[space_signature] = regressor

    def extract_features_and_labels(self, signature):
        features = [(t.space_sample.vectors, t.reward) for t in self.history.trials if
                    t.space_sample.signature == signature]
        return features