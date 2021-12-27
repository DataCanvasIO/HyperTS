import numpy as np
from hyperts.utils import consts
from hyperts.framework.dl import DeepAR, HybirdRNN
from ._base import EstimatorWrapper, WrapperMixin

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DeepARWrapper(EstimatorWrapper, WrapperMixin):

    def __init__(self, fit_kwargs, **kwargs):
        super(DeepARWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_dl_kwargs()
        self.model = DeepAR(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        y = self.fit_transform(y)
        self.model.fit(X, y, **self.fit_kwargs)

    def predict(self, X, **kwargs):
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            preds = self.model.forecast(X)
            preds = self.inverse_transform(preds)
            preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
            return preds
        else:
            return self.model.predict(X)


class HybirdRNNWrapper(EstimatorWrapper, WrapperMixin):

    def __init__(self, fit_kwargs, **kwargs):
        super(HybirdRNNWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_dl_kwargs()
        self.model = HybirdRNN(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        y = self.fit_transform(y)
        self.model.fit(X, y, **self.fit_kwargs)

    def predict(self, X, **kwargs):
        if self.init_kwargs.get('task') in consts.TASK_LIST_FORECAST:
            preds = self.model.forecast(X)
            preds = self.inverse_transform(preds)
            preds = np.clip(preds, a_min=1e-6, a_max=abs(preds)) if self.is_scale is not None else preds
            return preds
        else:
            return self.model.predict(X)