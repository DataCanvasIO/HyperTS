# -*- coding:utf-8 -*-

from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace, Choice
from hyperts.estimators import TSEstimatorMS, ProphetWrapper, VARWrapper, SKTimeWrapper


def search_space_univariate_forecast():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(ProphetWrapper, interval_width=Choice([0.5, 0.6]), seasonality_mode=Choice(['additive', 'multiplicative']))(input)
        space.set_inputs(input)
    return space


def search_space_multivariate_forecast():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(VARWrapper, ic=Choice(['aic', 'fpe', 'hqic', 'bic']))(input)
        space.set_inputs(input)
    return space


def space_classification_classification():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        TSEstimatorMS(SKTimeWrapper, n_estimators=Choice([50, 100, 150]))(input)
        space.set_inputs(input)
    return space


# TODO:  define others search space

