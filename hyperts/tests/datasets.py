# -*- coding:utf-8 -*-

import datetime
from random import random

import numpy as np
import pandas as pd


def get_random_univariate_forecast_dataset():
    X = pd.DataFrame({'ds': pd.date_range("20130101", periods=100)})
    y = pd.DataFrame({'value':  np.random.rand(1, 100)[0].tolist()})
    return X, y


def get_random_multivariate_forecast_dataset():
    now_date = datetime.datetime.now()
    # contrived dataset with dependency
    data = list()
    X = []
    for i in range(100):
        now_date = now_date + datetime.timedelta(days=1)
        X.append(now_date)
        v1 = i + random()
        v2 = v1 + random()
        row = [v1, v2]
        data.append(row)
    X = pd.DataFrame(data={'ds': X})
    y = pd.DataFrame(data=data)
    y.columns = ['var_1', 'var_2']
    return X, y


def get_random_univariate_forecast_dataset():
    X = pd.DataFrame({'ds': pd.date_range("20130101", periods=100)})
    y = pd.DataFrame({'value':  np.random.rand(1, 100)[0].tolist()})
    return X, y
