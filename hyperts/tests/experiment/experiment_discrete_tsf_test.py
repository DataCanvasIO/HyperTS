from hyperts import make_experiment
from hyperts.datasets import load_network_traffic
from hyperts.toolbox import temporal_train_test_split
from hyperts.tests import skip_if_not_tf, skip_if_not_prophet


class Test_Discrete_Time_Series():

    @skip_if_not_tf
    def test_univariate_discrete_forecast(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='dl', # only
                              task='forecast',
                              freq='null',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)

    @skip_if_not_tf
    def test_multivariate_discrete_forecast(self):
        df = load_network_traffic()
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='dl', # only
                              task='forecast',
                              freq='null',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)

    @skip_if_not_tf
    def test_univariate_cutoff_trainset_forecast(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='dl',
                              task='forecast',
                              forecast_train_data_periods=168*4,
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)

    @skip_if_not_prophet
    def test_multivariate_cutoff_trainset_forecast(self):
        df = load_network_traffic()
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='stats',
                              task='forecast',
                              forecast_train_data_periods=168*4,
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=2, final_train_epochs=2)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)