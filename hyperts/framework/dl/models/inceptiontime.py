# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from hyperts.utils import consts
from hyperts.framework.dl import layers
from hyperts.framework.dl import BaseDeepEstimator

from hypernets.utils import logging
logger = logging.get_logger(__name__)


def InceptionTimeModel():
    """

    """


class InceptionTime(BaseDeepEstimator):
    """

    """
    def __init__(self, **kwargs):
        super(InceptionTime, self).__init__(**kwargs)


    def _build_estimator(self, **kwargs):
        raise NotImplementedError('Return inception model.')

    def _fit(self, train_X, train_y, valid_X, valid_y, **kwargs):
        raise NotImplementedError('Return estimator.')

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, X):
        return self.model(X, training=False)