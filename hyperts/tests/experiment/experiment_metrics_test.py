from hyperts.datasets import load_arrow_head, load_fixed_univariate_forecast_dataset
from hyperts.utils import consts
from hyperts.utils import toolbox as dp
from hyperts.experiment import make_experiment, process_test_data


class Test_Univariable_Forecast_Metrics():
    def test_univariable_forecast_metrics_mse(self):
        _test_univariable_forecast_metric(consts.Metric_MSE)

    def test_univariable_forecast_metrics_rmse(self):
        _test_univariable_forecast_metric(consts.Metric_RMSE)

    def test_univariable_forecast_metrics_mae(self):
        _test_univariable_forecast_metric(consts.Metric_MAE)

    def test_univariable_forecast_metrics_mape(self):
        _test_univariable_forecast_metric(consts.Metric_MAPE)  # todo not supported

    def test_univariable_forecast_metrics_smape(self):
        _test_univariable_forecast_metric(consts.Metric_SMAPE)  # todo not supported

    def test_univariable_forecast_metrics_rmse(self):
        _test_univariable_forecast_metric(consts.Metric_RMSE)

    def test_univariable_forecast_metrics_r2(self):
        _test_univariable_forecast_metric(consts.Metric_R2)

    def test_univariable_forecast_metrics_msle(self):
        _test_univariable_forecast_metric(consts.Metric_MSLE)

    def test_univariable_forecast_metrics_None(self):
        _test_univariable_forecast_metric(None)


class Test_Univariable_BinaryClass_Metrics():
    def test_univariable_binaryclass_metrics_accuracy(self):
        _test_univariable_binaryclass_metric(consts.Metric_ACCURACY)

    def test_univariable_binaryclass_metrics_presicion(self):
        _test_univariable_binaryclass_metric(consts.Metric_PRESICION)

    def test_univariable_binaryclass_metrics_recall(self):
        _test_univariable_binaryclass_metric(consts.Metric_RECALL)

    def test_univariable_binaryclass_metrics_auc(self):
        _test_univariable_binaryclass_metric(consts.Metric_AUC)

    def test_univariable_binaryclass_metrics_f1(self):
        _test_univariable_binaryclass_metric(consts.Metric_F1)

    def test_univariable_binaryclass_metrics_logloss(self):
        _test_univariable_binaryclass_metric(consts.Metric_LOGLOSS)

    def test_univariable_binaryclass_metrics_None(self):
        _test_univariable_binaryclass_metric(None)


class Test_Univariable_MultiClass_Metrics():
    def test_univariable_multiclass_metrics_accuracy(self):
        _test_univariable_multiclass_metric(consts.Metric_ACCURACY)

    def test_univariable_multiclass_metrics_presicion(self):
        _test_univariable_multiclass_metric(consts.Metric_PRESICION)

    def test_univariable_multiclass_metrics_recall(self):
        _test_univariable_multiclass_metric(consts.Metric_RECALL)

    def test_univariable_multiclass_metrics_auc(self):
        _test_univariable_multiclass_metric(consts.Metric_AUC)

    def test_univariable_multiclass_metrics_f1(self):
        _test_univariable_multiclass_metric(consts.Metric_F1)

    def test_univariable_multiclass_metrics_logloss(self):
        _test_univariable_multiclass_metric(consts.Metric_LOGLOSS)

    def test_univariable_multiclass_metrics_none(self):
        _test_univariable_multiclass_metric(None)


def _test_univariable_forecast_metric(metric):
    def get_params_test_task():
        return "example_wp_log_peyton_manning.csv", {'timestamp': 'ds',
                                                     'optimize_direction': consts.OptimizeDirection_MINIMIZE,
                                                     'target': 'y'}

    reward_metric = metric
    task = consts.Task_FORECAST
    params = get_params_test_task()
    df = load_fixed_univariate_forecast_dataset()
    train_df, test_df = dp.temporal_train_test_split(df, test_size=0.1)
    timestamp = 'ds'
    exp = make_experiment(train_df, task=task, reward_metric=reward_metric, **params[1])
    model = exp.run(max_trials=1)
    X_test, y_test = process_test_data(test_df, timestamp=timestamp, impute=True)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape


def _test_univariable_binaryclass_metric(metric):
    df = load_arrow_head()
    df = df[df.target.isin(['0', '1'])]
    train_df, test_df = dp.random_train_test_split(df, test_size=0.2)

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

    X_test, y_test = process_test_data(test_df, target=target)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape


def _test_univariable_multiclass_metric(metric):
    df = load_arrow_head()
    train_df, test_df = dp.random_train_test_split(df, test_size=0.2)

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

    X_test, y_test = process_test_data(test_df, target=target)
    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape
