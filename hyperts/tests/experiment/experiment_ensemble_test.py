import pandas as pd

from hyperts import make_experiment
from hyperts.datasets import load_network_traffic, load_basic_motions, load_real_known_cause_dataset
from hyperts.toolbox import temporal_train_test_split, random_train_test_split
from hyperts.tests import skip_if_not_tf, skip_if_not_prophet


class Test_HyperTS_Ensemble():

    @skip_if_not_prophet
    def test_stats_univarite_forecast_nocv_ensemble(self):
        tsf_ensemble_test(univariate=True, mode='stats', cv=False, ensemble_size=3)

    @skip_if_not_prophet
    def test_stats_univarite_forecast_cv_ensemble(self):
        tsf_ensemble_test(univariate=True, mode='stats', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_univarite_forecast_nocv_ensemble(self):
        tsf_ensemble_test(univariate=True, mode='dl', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_univarite_forecast_cv_ensemble(self):
        tsf_ensemble_test(univariate=True, mode='dl', cv=True, ensemble_size=3)

    def test_stats_multivarite_forecast_nocv_ensemble(self):
        tsf_ensemble_test(univariate=False, mode='stats', cv=False, ensemble_size=3)

    def test_stats_multivarite_forecast_cv_ensemble(self):
        tsf_ensemble_test(univariate=False, mode='stats', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_multivarite_forecast_nocv_ensemble(self):
        tsf_ensemble_test(univariate=False, mode='dl', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_multivarite_forecast_cv_ensemble(self):
        tsf_ensemble_test(univariate=False, mode='dl', cv=True, ensemble_size=3)


    def test_stats_binary_classification_nocv_ensemble(self):
        tsc_ensemble_test(binary=True, mode='stats', cv=False, ensemble_size=3)

    def test_stats_binary_classification_cv_ensemble(self):
        tsc_ensemble_test(binary=True, mode='stats', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_binary_classification_nocv_ensemble(self):
        tsc_ensemble_test(binary=True, mode='dl', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_binary_classification_cv_ensemble(self):
        tsc_ensemble_test(binary=True, mode='dl', cv=True, ensemble_size=3)

    def test_stats_multi_classification_nocv_ensemble(self):
        tsc_ensemble_test(binary=False, mode='stats', cv=False, ensemble_size=3)

    def test_stats_multi_classification_cv_ensemble(self):
        tsc_ensemble_test(binary=False, mode='stats', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_multi_classification_nocv_ensemble(self):
        tsc_ensemble_test(binary=False, mode='dl', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_multi_classification_cv_ensemble(self):
        tsc_ensemble_test(binary=False, mode='dl', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_stats_anomaly_detection_nocv_ensemble(self):
        tsad_ensemble_test(mode='stats', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_stats_anomaly_detection_cv_ensemble(self):
        tsad_ensemble_test(mode='stats', cv=True, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_anomaly_detection_nocv_ensemble(self):
        tsad_ensemble_test(mode='dl', cv=False, ensemble_size=3)

    @skip_if_not_tf
    def test_dl_anomaly_detection_cv_ensemble(self):
        tsad_ensemble_test(mode='dl', cv=True, ensemble_size=3)


@skip_if_not_tf
@skip_if_not_prophet
def tsf_ensemble_test(univariate=True, mode='dl', cv=False, ensemble_size=None):
    df = load_network_traffic(univariate=False)
    if univariate:
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
    else:
        df = df.drop(columns=['Var_2', 'Var_4', 'Var_5', 'Var_6'])
    train_data, test_data = temporal_train_test_split(df, test_horizon=168)

    exp = make_experiment(train_data.copy(),
                            mode=mode,
                            cv=cv,
                            num_folds=3,
                            dl_gpu_usage_strategy=0,
                            task='forecast',
                            timestamp='TimeStamp',
                            covariables=['HourSin', 'WeekCos', 'CBWD'],
                            forecast_train_data_periods=24*14,
                            ensemble_size=ensemble_size,
                            max_trials=3,
                            random_state=202,
                            log_level='info')

    model = exp.run(epochs=1, final_train_epochs=2)

    X_test, y_test = model.split_X_y(test_data.copy())

    y_pred = model.predict(X_test)

    scores = model.evaluate(y_test, y_pred)

    print(scores)

@skip_if_not_tf
@skip_if_not_prophet
def tsc_ensemble_test(binary=False, mode='stats', cv=False, ensemble_size=None):
    df = load_basic_motions()
    if binary:
        df['target'] = df['target'].map(lambda x: x if x == 'standing' else 'notstanding')
    train_data, test_data = random_train_test_split(df, test_size=0.2, random_state=2022)

    experiment = make_experiment(train_data=train_data.copy(),
                                 task='classification',
                                 mode=mode,
                                 cv=cv,
                                 dl_gpu_usage_strategy=0, # GPU
                                 target='target',
                                 reward_metric='accuracy',
                                 max_trials=3,
                                 ensemble_size=ensemble_size,
                                 random_state=2022,
                                 log_level='info')

    model = experiment.run(epochs=1, final_train_epochs=2)

    X_test, y_test = model.split_X_y(test_data.copy())
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    scores = model.evaluate(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    print(scores)


@skip_if_not_tf
@skip_if_not_prophet
def tsad_ensemble_test(mode='stats', cv=False, ensemble_size=None):
    df = load_real_known_cause_dataset()
    df_ = df.drop(columns=['anomaly'])
    train_df, test_df = temporal_train_test_split(df_, test_horizon=15000)

    exp = make_experiment(train_data=train_df.copy(),
                          mode=mode,
                          task='detection',
                          timestamp='timestamp',
                          cv=cv,
                          ensemble_size=ensemble_size,
                          random_state=2022)

    model = exp.run(max_trials=3, epochs=10, final_train_epochs=10)

    X_test, _ = model.split_X_y(test_df.copy())
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    y_test = df.iloc[-15000:, 2]

    scores = model.evaluate(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    assert isinstance(scores, pd.DataFrame)
    assert y_pred.shape[0] == y_test.shape[0]
    print('univariate_anomaly_detection score: ', scores)