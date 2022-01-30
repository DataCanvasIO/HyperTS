import numpy as np
from hyperts.utils import consts, metrics, get_tool_box
from hyperts.experiment import make_experiment
from hyperts.datasets import load_arrow_head, load_fixed_univariate_forecast_dataset, load_network_traffic


class Test_Univariate_Forecast_Metrics():
    def test_univariate_forecast_metrics_mse(self):
        _test_univariate_forecast_metric(consts.Metric_MSE)

    def test_univariate_forecast_metrics_rmse(self):
        _test_univariate_forecast_metric(consts.Metric_RMSE)

    def test_univariate_forecast_metrics_mae(self):
        _test_univariate_forecast_metric(consts.Metric_MAE)

    def test_univariate_forecast_metrics_mape(self):
        _test_univariate_forecast_metric(consts.Metric_MAPE)

    def test_univariate_forecast_metrics_smape(self):
        _test_univariate_forecast_metric(consts.Metric_SMAPE)

    def test_univariate_forecast_metrics_r2(self):
        _test_univariate_forecast_metric(consts.Metric_R2)

    def test_univariate_forecast_metrics_msle(self):
        _test_univariate_forecast_metric(consts.Metric_MSLE)

    def test_univariate_forecast_metrics_None(self):
        _test_univariate_forecast_metric(None)


class Test_Univariate_BinaryClass_Metrics():
    def test_univariate_binaryclass_metrics_accuracy(self):
        _test_univariate_binaryclass_metric(consts.Metric_ACCURACY)

    def test_univariate_binaryclass_metrics_presicion(self):
        _test_univariate_binaryclass_metric(consts.Metric_PRESICION)

    def test_univariate_binaryclass_metrics_recall(self):
        _test_univariate_binaryclass_metric(consts.Metric_RECALL)

    def test_univariate_binaryclass_metrics_auc(self):
        _test_univariate_binaryclass_metric(consts.Metric_AUC)

    def test_univariate_binaryclass_metrics_f1(self):
        _test_univariate_binaryclass_metric(consts.Metric_F1)

    def test_univariate_binaryclass_metrics_logloss(self):
        _test_univariate_binaryclass_metric(consts.Metric_LOGLOSS)

    def test_univariate_binaryclass_metrics_None(self):
        _test_univariate_binaryclass_metric(None)


class Test_Univariate_MultiClass_Metrics():
    def test_univariate_multiclass_metrics_accuracy(self):
        _test_univariate_multiclass_metric(consts.Metric_ACCURACY)

    def test_univariate_multiclass_metrics_presicion(self):
        _test_univariate_multiclass_metric(consts.Metric_PRESICION)

    def test_univariate_multiclass_metrics_recall(self):
        _test_univariate_multiclass_metric(consts.Metric_RECALL)

    def test_univariate_multiclass_metrics_auc(self):
        _test_univariate_multiclass_metric(consts.Metric_AUC)

    def test_univariate_multiclass_metrics_f1(self):
        _test_univariate_multiclass_metric(consts.Metric_F1)

    def test_univariate_multiclass_metrics_logloss(self):
        _test_univariate_multiclass_metric(consts.Metric_LOGLOSS)

    def test_univariate_multiclass_metrics_none(self):
        _test_univariate_multiclass_metric(None)


class Test_Multivariate_Forecast_Metrics():
    def test_multivariate_forecast_metrics_mse(self):
        _test_multivariate_forecast(consts.Metric_MSE)

    def test_multivariate_forecast_metrics_rmse(self):
        _test_multivariate_forecast(consts.Metric_RMSE)

    def test_multivariate_forecast_metrics_mae(self):
        _test_multivariate_forecast(consts.Metric_MAE)

    def test_multivariate_forecast_metrics_mape(self):
        _test_multivariate_forecast(consts.Metric_MAPE)

    def test_multivariate_forecast_metrics_smape(self):
        _test_multivariate_forecast(consts.Metric_SMAPE)

    def test_multivariate_forecast_metrics_r2(self):
        _test_multivariate_forecast(consts.Metric_R2)

    def test_multivariate_forecast_metrics_msle(self):
        _test_multivariate_forecast(consts.Metric_MSLE)

    def test_multivariate_forecast_metrics_None(self):
        _test_multivariate_forecast(None)


def _test_univariate_forecast_metric(metric):
    def get_params_test_task():
        return "example_wp_log_peyton_manning.csv", {'timestamp': 'ds',
                                                     'optimize_direction': consts.OptimizeDirection_MINIMIZE,
                                                     'target': 'y'}

    reward_metric = metric
    task = consts.Task_FORECAST
    params = get_params_test_task()
    df = load_fixed_univariate_forecast_dataset()
    tb = get_tool_box(df)
    train_df, test_df = tb.temporal_train_test_split(df, test_size=0.1)
    exp = make_experiment(train_df, task=task, reward_metric=reward_metric, **params[1])
    model = exp.run(max_trials=1)
    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]


def _test_univariate_binaryclass_metric(metric):
    df = load_arrow_head()
    df = df[df.target.isin(['0', '1'])]
    tb = get_tool_box(df)
    train_df, test_df = tb.random_train_test_split(df, test_size=0.2)

    target = 'target'
    task = consts.Task_CLASSIFICATION
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MAXIMIZE

    params = {'pos_label': '1'}

    exp = make_experiment(train_df.copy(),
                          task=task,
                          eval_data=test_df.copy(),
                          target=target,
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction, **params)

    model = exp.run(max_trials=1)

    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)

    assert y_pred.shape[0] == y_test.shape[0]


def _test_univariate_multiclass_metric(metric):
    df = load_arrow_head()
    tb = get_tool_box(df)
    train_df, test_df = tb.random_train_test_split(df, test_size=0.2)

    target = 'target'
    task = consts.Task_CLASSIFICATION
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MAXIMIZE

    exp = make_experiment(train_df.copy(),
                          task=task,
                          eval_data=test_df.copy(),
                          target=target,
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction)

    model = exp.run(max_trials=1)

    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)

    assert y_pred.shape[0] == y_test.shape[0]


def _test_multivariate_forecast(metric):
    df = load_network_traffic()
    df.drop(['CBWD'], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    tb = get_tool_box(df)
    train_df, test_df = tb.temporal_train_test_split(df, test_size=0.1)

    timestamp = 'TimeStamp'
    task = consts.Task_MULTIVARIATE_FORECAST
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MINIMIZE

    exp = make_experiment(train_df,
                          timestamp=timestamp,
                          task=task,
                          callbacks=None,
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction)

    model = exp.run(max_trials=1)

    X_test, y_test = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    score = model.evaluate(y_test, y_pred)
    print('multivariate_forecast mape: ', score)
