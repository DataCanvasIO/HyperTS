import numpy as np
from hyperts.utils import consts
from hyperts.framework.wrappers import EstimatorWrapper
from hyperts.framework.wrappers import WrapperMixin

from hyperts.framework.dl import DeepAR
from hyperts.framework.dl import HybirdRNN
from hyperts.framework.dl import LSTNet
from hyperts.framework.dl import NBeats
from hyperts.framework.dl import InceptionTime
from hyperts.framework.dl import ConvVAE

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DeepARWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: univariate forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(DeepARWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = DeepAR(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.drop_sample_rate:
            X, y = self.drop_hist_sample(X, y, **self.init_kwargs)
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
            return self.model.meta.labels_
        else:
            return None


class HybirdRNNWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: forecast, classification and regression.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(HybirdRNNWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = HybirdRNN(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.drop_sample_rate:
            X, y = self.drop_hist_sample(X, y, **self.init_kwargs)
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
            return self.model.meta.labels_
        else:
            return None


class LSTNetWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: forecast, classification and regression.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(LSTNetWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = LSTNet(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.drop_sample_rate:
            X, y = self.drop_hist_sample(X, y, **self.init_kwargs)
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
            return self.model.meta.labels_
        else:
            return None


class NBeatsWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: forecast.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(NBeatsWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = NBeats(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.drop_sample_rate:
            X, y = self.drop_hist_sample(X, y, **self.init_kwargs)
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
            return self.model.meta.labels_
        else:
            return None


class InceptionTimeWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: classification.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(InceptionTimeWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = InceptionTime(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        fit_kwargs = self._merge_dict(self.fit_kwargs, kwargs)
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
            return self.model.meta.labels_
        else:
            return None


class ConvVAEWrapper(EstimatorWrapper, WrapperMixin):
    """
    Adapt: anomaly detection.
    """
    def __init__(self, fit_kwargs, **kwargs):
        kwargs = self.update_init_kwargs(**kwargs)
        super(ConvVAEWrapper, self).__init__(fit_kwargs, **kwargs)
        self.update_fit_kwargs()
        self.model = ConvVAE(**self.init_kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.drop_sample_rate:
            X, y = self.drop_hist_sample(X, y, **self.init_kwargs)
        TC, X = self.detection_split_XTC(X)
        X = self.fit_transform(X)
        kwargs['reconstract_dim'] = X.shape[1]
        fit_kwargs = self._merge_dict(self.fit_kwargs, kwargs)
        self.model.fit(TC, X, **fit_kwargs)
        if y is not None:
            self.y_unique_ = np.unique(y)

    def predict(self, X, **kwargs):
        TC, X = self.detection_split_XTC(X)
        X = self.transform(X)
        pred = self.model.predict_outliers(TC, X)
        if kwargs.get('return_confidence', False) is True:
            confidence = self.model.predict_outliers_confidence(TC, X)
            return pred, confidence
        return pred

    def predict_proba(self, X, **kwargs):
        TC, X = self.detection_split_XTC(X)
        X = self.transform(X)
        return self.model.predict_outliers_prob(TC, X)

    @property
    def classes_(self):
        return self.y_unique_