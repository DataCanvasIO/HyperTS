# Welcome to HyperTS

[简体中文](README_zh_CN.md)

[![Python Versions](https://img.shields.io/pypi/pyversions/hyperts.svg)](https://pypi.org/project/hyperts)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)](https://pypi.org/project/hyperts)
[![License](https://img.shields.io/github/license/DataCanvasIO/hyperts.svg)](https://github.com/DataCanvasIO/hyperts/blob/master/LICENSE)

:dizzy: Easy-to-use, powerful, unified full pipeline automated time series toolkit. Supports forecasting, classification and regression.


## We Are Hiring！
Dear folks, we are offering challenging opportunities located in Beijing for both professionals and students who are keen on AutoML/NAS. Come be a part of DataCanvas! Please send your CV to yangjian@zetyun.com. (Application deadline: TBD.) 


## Overview
HyperTS is a Python package that provides an end-to-end time series (TS) analysis toolkit. It covers complete and flexible AutoML workflows for TS, including data clearning, preprocessing, feature engineering, model selection, hyperparamter optimization, result evaluation, and visualization.

Multi-mode drive, light-heavy combination is the highlighted features of HyperTS. Therefore, statistical models (STATS), deep learning (DL), and neural architecture search (NAS) can be switched arbitrarily to get a powerful TS estimator.

As an easy-to-use and lower-thoreshold API, users can get a model after simply running the experiment, and then execute ```.predict()```, ```.predict_proba()```, ```.evalute()```, ```.plot()``` for various time series analysis.


## Tutorial

|[English Docs](https://hyperts.readthedocs.io/en/latest/) / [Chinese Docs](https://hyperts.readthedocs.io/zh_CN/latest)| Discription |
| --------------------------------- | --------------------------------- |
[Expected Data Format](https://hyperts.readthedocs.io/en/latest/contents/0300_dataformat.html)|What data formats do HyperTS expect?|
|[Quick Start](https://hyperts.readthedocs.io/en/latest/contents/0400_quick_start.html)| How to get started quickly with HyperTS?|
|[Advanced Ladder](https://hyperts.readthedocs.io/en/latest/contents/0500_advanced_config.html)|How to realize the potential of HyprTS?|
|[Custom Functions](https://hyperts.readthedocs.io/en/latest/contents/0600_custom_functions.html)|How to customize the functions of HyprTS?|

## Installation

HyperTS is available on Pypi and can be installed with ``pip``:

```bash
pip install hyperts
```

You can also install HyperTS from ``conda`` via the ``conda-forge`` channel:

```bash
conda install -c conda-forge hyperts
```

Note:

- Prophet is required by HyperTS, install it from ``conda`` before installing HyperTS using ``pip``.
- Tensorflow is an optional dependency for HyperTS, install it if using DL and NAS mode.

For more installation tips, see [installation](https://hyperts.readthedocs.io/en/latest/contents/0200_installation.html).

## Examples

Users can quickly create and ```run()``` an experiment with ```make_experiment()```, where ```train_data```, and ```task``` are required input parameters. In the following forecast example, we define the experiment as a multivariate-forecast ```task```, and use the statistical model (stat ```mode```) . Besides, the mandatory arguments ```timestamp``` and ```covariables``` (if have) should also be defined in the experiment.

```python
from hyperts import make_experiment
from hyperts.datasets import load_network_traffic

from sklearn.model_selection import train_test_split

data = load_network_traffic()
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model = make_experiment(train_data.copy(),
                        task='multivariate-forecast',
                        mode='stats',
                        timestamp='TimeStamp',
                        covariates=['HourSin', 'WeekCos', 'CBWD']).run()

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

**Multi-mode Support:** A large collection of TS models, including statistical models (Prophet,ARIMA,VAR,etc.) , deep learning models(DeepAR,GRU,LSTNet,etc), and neural architecture search (to be developed).

**Multi-variate Support:** From univariate to multivariate time series.

**Covariates Support:** Deep learning models support covariates as input featues for time series forecasting. 

**Probabilistic Intervals Support:** Time series forecsting visualization can show confidence intervals.

**Abundant Metrics:** A variety of performance metrics to evaluate results and guide models optimization, including MSE, SMAPE, Accuracy, F1-Score,etc. 

**Powerful Search Strategies:** Adapting Grid Search, Monte Carlo Tree Search, Evolution Algorithm combined with a meta-learner to enable a powerful and effective TS pipeline.

**Ensemble:** Refinement greedy emsemble glues the most powerful models.

**Cross Validation:** Multiple time series cross-validation strategies ensure generalization ability.

## Communication
If you wish to contribute to this project, please refer to [CONTRIBUTING](CONTRIBUTING.md).

## HyperTS Related Projects
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): A general automated machine learning (AutoML) framework.
* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): A full pipeline AutoML tool integrated various GBM models.
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): An AutoDL tool for tabular data.
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): An AutoDL tool for Neural Architecture Search and Hyperparameter Optimization on Tensorflow and Keras.
* [HyperBoard](https://github.com/DataCanvasIO/HyperBoard): A visualization tool for Hypernets.
* [Cooka](https://github.com/DataCanvasIO/Cooka): Lightweight interactive AutoML system.

![DataCanvas AutoML Toolkit](docs/static/images/datacanvas_automl_toolkit.png)

## DataCanvas

![datacanvas](docs/static/images/dc_logo_1.png)

HyperTS is an open source project created by [DataCanvas](https://www.datacanvas.com/). 