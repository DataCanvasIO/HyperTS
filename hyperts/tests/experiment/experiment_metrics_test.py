import pandas as pd

from hyperts.datasets import load_arrow_head
from hyperts.utils import consts
from hyperts.utils import toolbox as dp
from hyperts.experiment import make_experiment, process_test_data


class Test_Experiment_Forecast_Metrics():
    def test_task_forecast_metrics_mse(self):
        _test_task_forecast_metric(consts.Metric_MSE)

    def test_task_forecast_metrics_rmse(self):
        _test_task_forecast_metric(consts.Metric_RMSE)

    def test_task_forecast_metrics_mae(self):
        _test_task_forecast_metric(consts.Metric_MAE)

    def test_task_forecast_metrics_mape(self):
        _test_task_forecast_metric(consts.Metric_MAPE)  # todo not supported

    def test_task_forecast_metrics_smape(self):
        _test_task_forecast_metric(consts.Metric_SMAPE)  # todo not supported

    def test_task_forecast_metrics_rmse(self):
        _test_task_forecast_metric(consts.Metric_RMSE)

    def test_task_forecast_metrics_r2(self):
        _test_task_forecast_metric(consts.Metric_R2)

    def test_task_forecast_metrics_msle(self):
        _test_task_forecast_metric(consts.Metric_MSLE)


class Test_Experiment_Classification_Metrics():
    def test_task_classification_metrics_accuracy(self):
        _test_task_classification_metric(consts.Metric_ACCURACY)

    def test_task_classification_metrics_presicion(self):
        _test_task_classification_metric(consts.Metric_PRESICION)

    def test_task_classification_metrics_recall(self):
        _test_task_classification_metric(consts.Metric_RECALL)

    def test_task_classification_metrics_auc(self):
        _test_task_classification_metric(consts.Metric_AUC)

    def test_task_classification_metrics_f1(self):
        _test_task_classification_metric(consts.Metric_F1)

    def test_task_classification_metrics_logloss(self):
        _test_task_classification_metric(consts.Metric_LOGLOSS)


def _test_task_forecast_metric(metric):
    def get_params_test_task():
        return "example_wp_log_peyton_manning.csv", {'timestamp': 'ds',
                                                     'optimize_direction': consts.OptimizeDirection_MINIMIZE,
                                                     'target': 'y'}

    reward_metric = metric
    task = consts.Task_FORECAST
    params = get_params_test_task()
    df = pd.read_csv("../../datasets/example_wp_log_peyton_manning.csv")
    train_df, test_df = dp.temporal_train_test_split(df, test_size=0.1)
    timestamp = 'ds'
    exp = make_experiment(train_df, task=task, reward_metric=reward_metric, **params[1])
    model = exp.run(max_trials=1)
    X_test, y_test = process_test_data(test_df, timestamp=timestamp, impute=True)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape


def _test_task_classification_metric(metric):
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
