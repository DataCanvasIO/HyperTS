# -*- coding:utf-8 -*-

from hyperts.hyper_ts import TSEstimatorMS, ProphetWrapper, VARWrapper
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def ts_stats_search_space():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(ProphetWrapper, interval_width=Choice([0.5, 0.6, 0.7, 0.8]), seasonality_mode=Choice(['additive', 'multiplicative']))(input)
        space.set_inputs(input)
    return space


#
def ts_multivariate_stats_search_space():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(VARWrapper, ic=Choice(['aic', 'fpe', 'hqic', 'bic']))(input)
        space.set_inputs(input)
    return space

# todo define a deepts search space



