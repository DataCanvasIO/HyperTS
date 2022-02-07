# HyperTS
Easy-to-use, powerful, unified full pipeline automated time series toolkit. Supports forecasting, classification, regression and others.

## Examples

```python
from hyperts.experiment import make_experiment
from hyperts.datasets import load_network_traffic

from sklearn.model_selection import train_test_split

data = load_network_traffic()
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model = make_experiment(train_data.copy(),
                        task='forecast',
                        mode='dl',
                        forecast_window=24*2,
                        timestamp='TimeStamp',
                        covariables=['HourSin', 'WeekCos', 'CBWD'],
                        max_trials=100).run()

X_test, y_test = model.split_X_y(test_data.copy())

y_pred = model.predict(X_test)

scores = model.evaluate(y_test, y_pred)

model.plot(forecast=y_pred, actual=test_data, var_id=0)
```

## DataCanvas

![datacanvas](docs/static/images/dc_logo_1.png)

HyperTS is an open source project created by [DataCanvas](https://www.datacanvas.com/). 