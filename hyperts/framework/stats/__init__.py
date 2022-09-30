"""
Supported models: Prophet, Arima, VAR, TSForest, KNeighbors.
"""
from ._base import BaseAnomalyDetector
from .iforest import TSIsolationForest