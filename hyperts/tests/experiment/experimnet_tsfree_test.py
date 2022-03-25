from hyperts import make_experiment
from hyperts.datasets import load_network_traffic
from hyperts.toolbox import temporal_train_test_split

class Test_HyperTS_TimeStamp_Free():

    def test_stats_forecast_tsfree(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6', 'TimeStamp'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='stats',
                              cv=True,
                              num_folds=3,
                              task='forecast',
                              timestamp='null',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)

    def test_dl_forecast_tsfree(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6', 'TimeStamp'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='dl',
                              dl_gpu_usage_strategy=0,
                              cv=True,
                              num_folds=3,
                              task='forecast',
                              timestamp='null',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)