"""
Supported models: Prophet, Arima, VAR, TSForest, KNeighbors.
"""
from .iforest import TSIsolationForest
from .ocsvm import TSOneClassSVM
from .tcforest import TimeSeriesForestClassifier
from .tctde import IndividualTDEClassifier