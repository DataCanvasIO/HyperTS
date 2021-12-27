import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from hyperts.datasets import *
from hyperts.utils.metrics import rmse, mape
from hyperts.utils import toolbox as tstb, consts
from hyperts.framework.wrappers import DeepARWrapper, HybirdRNNWrapper



class Test_DL_Wrappers():

    def test_univariate_forecast_deepar(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        X = tstb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tstb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIABLE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'lstm',
            'nb_units': 10,
            'nb_layers': 2,

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = DeepARWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse > 0
        assert score_mape > 0


    def test_univariate_forecast_rnn(self):
        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        X = tstb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tstb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIABLE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'simple_rnn',
            'nb_units': 10,
            'nb_layers': 2,
            'learning_rate': 0.001,
            'loss': 'mse',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse > 0
        assert score_mape > 0


    def test_multivariate_forecast_with_covariables_rnn(self):
        X, y = load_network_traffic(return_X_y=True)
        y = tstb.simple_numerical_imputer(y)
        X_train, X_test, y_train, y_test = tstb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIABLE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'lstm',
            'nb_units': 10,
            'nb_layers': 2,
            'learning_rate': 0.001,
            'loss': 'mse',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse > 0
        assert score_mape > 0


    def test_multivariate_forecast_no_covariables_rnn(self):
        X, y = load_network_traffic(return_X_y=True)
        y = tstb.simple_numerical_imputer(y)
        X = X[['TimeStamp']]
        X_train, X_test, y_train, y_test = tstb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIABLE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'gru',
            'nb_units': 10,
            'nb_layers': 2,
            'learning_rate': 0.001,
            'loss': 'mse',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse > 0
        assert score_mape > 0