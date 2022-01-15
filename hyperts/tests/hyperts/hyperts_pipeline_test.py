from hyperts import HyperTS

from hyperts.utils import consts, get_tool_box
from hyperts.datasets import load_network_traffic

class Test_HyperTS_Pipeline:

    def test_forecast_HyperTS(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'], axis=1)

        tb = get_tool_box(df)
        train_df, test_df = tb.temporal_train_test_split(df, test_horizon=168)

        config = {
            'timestamp': 'TimeStamp',
            'covariables': ['HourSin', 'WeekCos', 'CBWD'],
            'task': consts.Task_FORECAST,
            'reward_metric': consts.Metric_MAE,
            'optimize_direction': consts.OptimizeDirection_MINIMIZE,
            'log_level': None
        }

        model = HyperTS(train_data=train_df, **config)

        model.fit(max_trials=3)

        X_test, y_test = model.split_X_y(test_df)
        forecast = model.predict(X_test)

        model.evaluate(y_test, forecast)