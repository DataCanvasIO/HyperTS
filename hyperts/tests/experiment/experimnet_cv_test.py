from hyperts import make_experiment
from hyperts.datasets import load_network_traffic, load_basic_motions
from hyperts.toolbox import temporal_train_test_split, random_train_test_split

class Test_HyperTS_Cross_Validation():

    def test_stats_forecast_cv(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='stats',
                              cv=True,
                              num_folds=3,
                              task='forecast',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)

    def test_dl_forecast_cv(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        exp = make_experiment(train_data.copy(),
                              mode='dl',
                              dl_gpu_usage_strategy=0,
                              cv=True,
                              num_folds=3,
                              task='forecast',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'],
                              max_trials=3,
                              random_state=2022)
        model = exp.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())

        y_pred = model.predict(X_test)

        assert y_pred.shape[0] == y_test.shape[0]

        scores = model.evaluate(y_test, y_pred)

        print(scores)


    def test_stats_classification_cv(self):
        df = load_basic_motions()
        train_data, test_data = random_train_test_split(df, test_size=0.2, random_state=2022)

        experiment = make_experiment(train_data=train_data.copy(),
                                     task='classification',
                                     mode='stats',
                                     cv=True,
                                     target='target',
                                     reward_metric='accuracy',
                                     max_trials=3,
                                     random_state=2022)

        model = experiment.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        scores = model.evaluate(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

        print(scores)

    def test_dl_classification_cv(self):
        df = load_basic_motions()
        train_data, test_data = random_train_test_split(df, test_size=0.2, random_state=2022)

        experiment = make_experiment(train_data=train_data.copy(),
                                     task='classification',
                                     mode='dl',
                                     dl_gpu_usage_strategy=0,
                                     cv=True,
                                     target='target',
                                     reward_metric='accuracy',
                                     max_trials=3,
                                     random_state=2022)

        model = experiment.run(epochs=20)

        X_test, y_test = model.split_X_y(test_data.copy())
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        scores = model.evaluate(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

        print(scores)