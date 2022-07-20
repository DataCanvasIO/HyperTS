import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from hyperts import make_experiment
from hyperts.datasets import load_network_traffic, load_basic_motions
from hyperts.toolbox import temporal_train_test_split, random_train_test_split

from hyperts.framework.search_space import Choice, Real, Int,\
                                           StatsForecastSearchSpace, \
                                           DLForecastSearchSpace, \
                                           StatsClassificationSearchSpace, \
                                           DLClassRegressSearchSpace
from hyperts.tests import skip_if_not_tf, skip_if_not_prophet


class Test_User_Defined_Search_Space():

    @skip_if_not_prophet
    def test_stats_forecast(self):
        df = load_network_traffic(univariate=False)
        df = df.drop(columns=['Var_1', 'Var_2', 'Var_4', 'Var_5', 'Var_6'])
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        my_stats_search_space = StatsForecastSearchSpace(
            var_init_kwargs=False,
            prophet_init_kwargs={
                'changepoint_range': Real(low=0.8, high=0.9, step=0.2)
            }
        )

        experiment = make_experiment(train_data.copy(),
                              mode='stats',
                              task='forecast',
                              timestamp='TimeStamp',
                              covariates=['HourSin', 'WeekCos', 'CBWD'],
                              cv=False,
                              ensemble_size=None,
                              search_space=my_stats_search_space,
                              forecast_train_data_periods=24 * 7,
                              max_trials=3,
                              random_state=2022)

        model = experiment.run()
        pipeline_params = model.get_pipeline_params()
        best_trial_params = experiment.report_best_trial_params()

        print(pipeline_params)
        print(best_trial_params)

    @skip_if_not_tf
    def test_dl_forecast(self):
        df = load_network_traffic(univariate=False)
        train_data, test_data = temporal_train_test_split(df, test_horizon=168)

        my_dl_search_space = DLForecastSearchSpace(
            enable_lstnet=False,
            hybirdrnn_init_kwargs={
                'rnn_type': Choice(['gru', 'lstm', 'simple_rnn']),
                'rnn_units': Int(low=32, high=128, step=16),
            }
        )

        experiment = make_experiment(train_data.copy(),
                                     mode='dl',
                                     task='forecast',
                                     timestamp='TimeStamp',
                                     covariates=['HourSin', 'WeekCos', 'CBWD'],
                                     cv=False,
                                     ensemble_size=None,
                                     search_space=my_dl_search_space,
                                     forecast_train_data_periods=24 * 7,
                                     max_trials=3,
                                     random_state=2022)

        model = experiment.run(epochs=2, final_train_epochs=2)
        pipeline_params = model.get_pipeline_params()
        best_trial_params = experiment.report_best_trial_params()

        print(pipeline_params)
        print(best_trial_params)

    def test_stats_classification(self):
        df = load_basic_motions()
        df['target'] = df['target'].map(lambda x: x if x == 'standing' else 'notstanding')
        train_data, test_data = random_train_test_split(df, test_size=0.2, random_state=2022)

        my_stats_search_space = StatsClassificationSearchSpace(
            knn_init_kwargs={
                'n_neighbors': Choice([2, 3, 4]),
            }
        )

        experiment = make_experiment(train_data=train_data.copy(),
                                     task='classification',
                                     mode='stats',
                                     cv=False,
                                     ensemble_size=None,
                                     pos_label='standing',
                                     target='target',
                                     reward_metric='f1',
                                     search_space=my_stats_search_space,
                                     max_trials=3,
                                     random_state=2022)

        model = experiment.run()
        pipeline_params = model.get_pipeline_params()
        best_trial_params = experiment.report_best_trial_params()

        print(pipeline_params)
        print(best_trial_params)

    @skip_if_not_tf
    def test_dl_classification(self):
        df = load_basic_motions()
        train_data, test_data = random_train_test_split(df, test_size=0.2, random_state=2022)

        my_dl_search_space = DLClassRegressSearchSpace(
            hybirdrnn_init_kwargs={
                'rnn_units': Choice([32, 64, 128]),
            },

            lstnet_init_kwargs={
                'kernel_size': Choice([3, 6]),
            }
        )

        experiment = make_experiment(train_data=train_data.copy(),
                                     task='classification',
                                     mode='dl',
                                     cv=False,
                                     ensemble_size=None,
                                     target='target',
                                     reward_metric='accuracy',
                                     search_space=my_dl_search_space,
                                     max_trials=3,
                                     random_state=2022)

        model = experiment.run(epochs=2, final_train_epochs=2)
        pipeline_params = model.get_pipeline_params()
        best_trial_params = experiment.report_best_trial_params()

        print(pipeline_params)
        print(best_trial_params)

    @skip_if_not_tf
    def test_nas_search_space(self):
        import tensorflow as tf
        from hypernets.searchers.random_searcher import RandomSearcher
        from hyperts.framework.dl import layers
        from hyperts.framework.search_space.micro_search_space import TSNASGenrealSearchSpace

        sfss = TSNASGenrealSearchSpace()
        sfss.update_init_params(
            task='task',
            timestamp='timestamp',
            metrics='acc',
            covariables=None,
            window=[3, 6, 9],
            horizon=1,
            freq='H'
        )

        searcher = RandomSearcher(sfss, optimize_direction='min')
        space_sample = searcher.sample()
        inp1 = layers.Input(shape=(4, 3))
        inp2 = layers.Input(shape=(4, 5))
        concat = layers.Concatenate()([inp1, inp2])
        space, outputs = space_sample.compile_and_forward(concat)
        fit_params = space_sample.__dict__.get('hyperparams').param_values
        print(fit_params)

        model = tf.keras.models.Model(inputs=[inp1, inp2], outputs=outputs[0])
        model.summary()