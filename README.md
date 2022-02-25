# Welcome to HyperTS

[![Python Versions](https://img.shields.io/pypi/pyversions/hypernets.svg)](https://pypi.org/project/hypernets)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)](https://pypi.org/project/deeptables)
[![License](https://img.shields.io/github/license/DataCanvasIO/deeptables.svg)](https://github.com/DataCanvasIO/deeptables/blob/master/LICENSE)

:dizzy: Easy-to-use, powerful, unified full pipeline automated time series toolkit. Supports forecasting, and regression.


## We Are Hiring！
Dear folks, we are offering challenging opportunities located in Beijing for both professionals and students who are keen on AutoML/NAS. Come be a part of DataCanvas! Please send your CV to yangjian@zetyun.com. (Application deadline: TBD.) 


## Overview
HyperTS is a Python package that provides an end-to-end time series (TS) analysis toolkit. It covers complete and flexible AutoML workflows for TS, including data clearning, preprocessing, feature engineering, model selection, hyperparamter optimization, result evaluation, and visualization.

Multi-mode drive, light-heavy combination is the highlight feature of HyperTS. Therefore, statistical models (STATS), deep learning (DL), and neural architecture search (NAS) can be switched arbitrarily to get a powerful TS estimator.

Easy-to-use and lower-level API. Users can get a model after simply running the experiment, and then execute ```.predict()```, ```.predict_proba()```, ```.evalute()```, ```.plot()``` for various time series analysis.


## Tutorial

|[English Docs](https://hyperts.readthedocs.io/en/latest/) | [Chinese Docs](https://hyperts.readthedocs.io/zh_CN/latest)|
| --------------------------------- | --------------------------------- |
[Expected Data Format](https://hyperts.readthedocs.io/en/latest/contents/0300_dataformat.html)|What data formats do HyperTS expect?|
|[Quick Start](https://hyperts.readthedocs.io/en/latest/contents/0400_quick_start.html)| How to get started quickly with HyperTS?|
|[Advanced Ladder](https://hyperts.readthedocs.io/en/latest/contents/0500_examples.html)|How to realize the potential of HyprTS?|
|[Custom Functions](https://hyperts.readthedocs.io/en/latest/contents/0600_custom_functions.html)|How to customize the functions of HyprTS?|


## Examples

Users can quickly create and ```run()``` an experiment with ```make_experiment()```, where ```train_data```, and ```task``` as required input parameters. In the following forecast example, we tell the experiment that it is a multivariate-forecast ```task```, using stats ```mode```, since the data contains timestamp and covariable columns, ```timestamp``` and ```covariables``` parameters might inform the experiment.

```python
from hyperts.experiment import make_experiment
from hyperts.datasets import load_network_traffic

from sklearn.model_selection import train_test_split

data = load_network_traffic()
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model = make_experiment(train_data.copy(),
                        task='multivariate-forecast',
                        mode='stats',
                        timestamp='TimeStamp',
                        covariables=['HourSin', 'WeekCos', 'CBWD']).run()

X_test, y_test = model.split_X_y(test_data.copy())

y_pred = model.predict(X_test)

scores = model.evaluate(y_test, y_pred)

model.plot(forecast=y_pred, actual=test_data)
```

![Forecast_Figure](docs/static/images/Actual_vs_Forecast.jpg)

- More detailed guides: [EXAMPLES.](https://github.com/DataCanvasIO/HyperTS/tree/main/examples)

## Key Features

HyperTS supports the following features:

**Multi-task Support:** Time series forecasting, classification, and regression.

**Multi-mode Support:** A large collection of TS models, from statistical models to deep learning models, and to neural architecture search (developing).

**Multi-variate Support:** From univariate to multivariate time series.

**Covariates Support:** Deep learning models support covariates as input featues for time series forecasting. 

**Probabilistic intervals Support:** Time series forecsting visualization can show confidence intervals.

**Abundant Metrics:** From MSE, SMAPE, Accuracy to F1-Score, a variety of performance metrics to evaluate results and guide models optimization.

**Powerful search strategies:** Adapting Grid Search, Monte Carlo Tree Search, Evolution Algorithm combined with a meta-learner to learn a powerful and effective TS pipeline.

## Communication
If you wish to contribute to this project, please refer to [CONTRIBUTING](CONTRIBUTING.md).

## HyperTS Related Projects
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): A general automated machine learning (AutoML) framework.
* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): A full pipeline AutoML tool integrated various GBM models.
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): An AutoDL tool for tabular data.
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): An AutoDL tool for Neural Architecture Search and Hyperparameter Optimization on Tensorflow and Keras.
* [Cooka](https://github.com/DataCanvasIO/Cooka): Lightweight interactive AutoML system.

![DataCanvas AutoML Toolkit](docs/static/images/datacanvas_automl_toolkit.png)

## DataCanvas

![datacanvas](docs/static/images/dc_logo_1.png)

HyperTS is an open source project created by [DataCanvas](https://www.datacanvas.com/). 