# Welcome to HyperTS

Easy-to-use, powerful, unified full pipeline automated time series toolkit. Supports forecasting, classification, regression and others.

## Overview
HyperTS is a Python package that provides an end-to-end time series (TS) analysis toolkit. It covers complete and flexible AutoML workflows for TS, including data clearning, preprocessing, feature engineering, model selection, hyperparamter optimization, result evaluation, and visualization.

Multi-mode drive, light-heavy combination is the highlight feature of HyperTS. Therefore, statistical models (STATS), deep learning (DL), and neural architecture search (NAS) can be switched arbitrarily to get a powerful TS estimator.

Easy-to-use and lower-level API. Users can get a model after simply running the experiment, and then execute ```.predict()```, ```.predict_proba()```, ```.evalute()```, ```.plot()``` for various time series analysis.


## Examples

Users can quickly create and ```run()``` an experiment with ```make_experiment()```, where ```train_data```, and ```task``` as required input parameters. In the forecast task in the following example, we tell the experiment that it is a multivariate-forecast ```task```, using stats ```mode```, since the data contains timestamp and covariable columns, ```timestamp``` and ```covariables``` parameters might inform the experiment.

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

model.plot(forecast=y_pred, actual=test_data, var_id=0)
```

<div align="center"><img src="docs/static/images/Actual_vs_Forecast.jpg" width="800"/></div>

- More detailed guides: [EXAMPLES.](https://github.com/DataCanvasIO/HyperTS/tree/main/examples)

## Key Features

HyperTS supports the following features:

**Multi-task Support:** Time series forecasting, classification, and regression.

**Multi-mode Support:** A large collection of TS models, from statistical models to deep learning models, and to neural architecture search (developing).

**Multi-variate Support:** From univariate to multivariate time series.

**Covariates Support:** Deep learning models support covariates as input featues for time series forecasting. 

**Probabilistic intervals Support:** Time series forecsting visualization can show confidence intervals.

**Abundant Metrics:** From ```MSE```,``` SMAPE```, ```Accuracy``` to ```F1-Score```, a variety of performance metrics to evaluate results and guide models optimization.

**Powerful search strategies:** Adapting Grid Search, Monte Carlo Tree Search, Evolution Algorithm combined with a meta-learner to learn a powerful and effective pipeline.

## Communication
If you wish to contribute to this project, please refer to [CONTRIBUTING](CONTRIBUTING.md).

## DataCanvas

![datacanvas](docs/static/images/dc_logo_1.png)

HyperTS is an open source project created by [DataCanvas](https://www.datacanvas.com/). 