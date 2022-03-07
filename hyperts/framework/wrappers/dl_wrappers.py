import numpy as np
from hyperts.utils import consts
from hyperts.framework.dl import DeepAR, HybirdRNN, LSTNet
from hyperts.framework.wrappers._base import EstimatorWrapper, WrapperMixin

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DeepARWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(DeepARWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_dl_kwargs()
        self.model = DeepAR(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        fit_kwargs = self._merge_dict(self.fit_kwargs, kwargs)
        y = self.fit_transform(y)
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            preds = self.model.forecast(X)
            preds = self.inverse_transform(preds)
            preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
            return preds
        else:
            X = self.transform(X)
            return self.model.predict_proba(X)

    @property
    def classes_(self):
        if self.init_kwargs.get('task') in consts.TASK_LIST_CLASSIFICATION:
            return self.model.mata.labels_
        else:
            return None


class HybirdRNNWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: forecast, classification and regression.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(HybirdRNNWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_dl_kwargs()
        self.model = HybirdRNN(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        fit_kwargs = self._merge_dict(self.fit_kwargs, kwargs)
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            y = self.fit_transform(y)
        else:
            X = self.fit_transform(X)
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            preds = self.model.forecast(X)
            preds = self.inverse_transform(preds)
            preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
            return preds
        elif self.init_kwargs.get('task') in consts.TASK_LIST_CLASSIFICATION:
            X = self.transform(X)
            return self.model.predict(X)
        else:
            X = self.transform(X)
            return self.model.predict_proba(X)

    def predict_proba(self, X, **kwargs):
        X = self.transform(X)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        if self.init_kwargs.get('task') in consts.TASK_LIST_CLASSIFICATION:
            return self.model.mata.labels_
        else:
            return None


class LSTNetWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: forecast, classification and regression.
    """
    def __init__(self, fit_kwargs, **kwargs):
        super(LSTNetWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_dl_kwargs()
        self.model = LSTNet(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        fit_kwargs = self._merge_dict(self.fit_kwargs, kwargs)
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            y = self.fit_transform(y)
        else:
            X = self.fit_transform(X)
        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            preds = self.model.forecast(X)
            preds = self.inverse_transform(preds)
            preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
            return preds
        elif self.init_kwargs.get('task') in consts.TASK_LIST_CLASSIFICATION:
            X = self.transform(X)
            return self.model.predict(X)
        else:
            X = self.transform(X)
            return self.model.predict_proba(X)

    def predict_proba(self, X, **kwargs):
        X = self.transform(X)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        if self.init_kwargs.get('task') in consts.TASK_LIST_CLASSIFICATION:
            return self.model.mata.labels_
        else:
            return None