import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from hyperts.datasets import *
from hyperts.utils.metrics import rmse, mape, accuracy_score, f1_score
from hyperts.utils import consts
from hyperts.utils import get_tool_box
from hyperts.tests import skip_if_not_tf

@skip_if_not_tf
class Test_DL_Wrappers():

    def test_univariate_forecast_deepar(self):
        from hyperts.framework.wrappers.dl_wrappers import DeepARWrapper

        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X = tb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'lstm',
            'rnn_units': 10,
            'rnn_layers': 2,

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = DeepARWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_univariate_forecast_rnn(self):
        from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper

        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X = tb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'basic',
            'rnn_units': 10,
            'rnn_layers': 2,
            'learning_rate': 0.001,
            'loss': 'mae',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_with_covariables_rnn(self):
        from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'lstm',
            'rnn_units': 10,
            'rnn_layers': 2,
            'learning_rate': 0.001,
            'loss': 'mae',
            'out_activation': 'sigmoid',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_no_covariables_rnn(self):
        from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X = X[['TimeStamp']]
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'gru',
            'rnn_units': 10,
            'rnn_layers': 2,
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

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_univariate_classification_rnn(self):
        from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper

        X, y = load_arrow_head(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,

            'rnn_type': 'basic',
            'rnn_units': 10,
            'rnn_layers': 3,
            'learning_rate': 0.001,
            'reducelr_patience': 15,
            'earlystop_patience': 20,

            'x_scale': np.random.choice(['z_score', 'scale-none'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_multivariate_classification_rnn(self):
        from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper

        X, y = load_basic_motions(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,

            'rnn_type': 'basic',
            'rnn_units': 10,
            'rnn_layers': 2,
            'learning_rate': 0.001,

            'x_scale': np.random.choice(['z_score', 'scale-none'], size=1)[0]
        }
        model = HybirdRNNWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_univariate_forecast_lstnet(self):
        from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper

        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X = tb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'basic',
            'skip_rnn_type': 'basic',
            'cnn_filters': 16,
            'kernel_size': 3,
            'rnn_units': 10,
            'rnn_layers': 2,
            'skip_rnn_units': 10,
            'skip_rnn_layers': 2,
            'skip_period': 3,
            'ar_order': 3,
            'learning_rate': 0.001,
            'loss': 'mae',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = LSTNetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_with_covariables_lstnet(self):
        from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'basic',
            'skip_rnn_type': 'basic',
            'cnn_filters': 16,
            'kernel_size': 3,
            'rnn_units': 10,
            'rnn_layers': 2,
            'skip_rnn_units': 10,
            'skip_rnn_layers': 2,
            'skip_period': 3,
            'ar_order': 3,
            'learning_rate': 0.001,
            'loss': 'mae',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = LSTNetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_no_covariables_lstnet(self):
        from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X = X[['TimeStamp']]
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,

            'rnn_type': 'basic',
            'skip_rnn_type': 'basic',
            'cnn_filters': 16,
            'kernel_size': 3,
            'rnn_units': 10,
            'rnn_layers': 2,
            'skip_rnn_units': 10,
            'skip_rnn_layers': 2,
            'skip_period': 3,
            'ar_order': 3,
            'learning_rate': 0.001,
            'loss': 'mae',

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = LSTNetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_univariate_classification_lstnet(self):
        from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper

        X, y = load_arrow_head(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,

            'rnn_type': 'gru',
            'rnn_units': 10,
            'rnn_layers': 2,
            'learning_rate': 0.001,

            'x_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0],
        }

        model = LSTNetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_multivariate_classification_lstnet(self):
        from hyperts.framework.wrappers.dl_wrappers import LSTNetWrapper

        X, y = load_basic_motions(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 100,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,

            'rnn_type': 'lstm',
            'skip_rnn_type': 'gru',
            'cnn_filters': 16,
            'kernel_size': 3,
            'rnn_units': 10,
            'rnn_layers': 2,
            'skip_rnn_units': 10,
            'skip_rnn_layers': 2,
            'skip_period': 0,
            'ar_order': 0,
            'learning_rate': 0.001,

            'x_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0],

        }
        model = LSTNetWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_univariate_forecast_nbeats(self):
        from hyperts.framework.wrappers.dl_wrappers import NBeatsWrapper

        X, y = load_random_univariate_forecast_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X = tb.simple_numerical_imputer(X, mode='mode')
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_FORECAST
        timestamp = 'ds'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,
            'forecast_length': 3,
            'hidden_layer_units': 32,

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = NBeatsWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_no_covariables_nbeats(self):
        from hyperts.framework.wrappers.dl_wrappers import NBeatsWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X = X[['TimeStamp']]
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,
            'forecast_length': 3,
            'hidden_layer_units': 32,

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = NBeatsWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_multivariate_forecast_with_covariables_nbeats(self):
        from hyperts.framework.wrappers.dl_wrappers import NBeatsWrapper

        X, y = load_network_traffic(return_X_y=True)
        tb = get_tool_box(X)
        y = tb.simple_numerical_imputer(y)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_FORECAST
        timestamp = 'TimeStamp'

        fit_kwargs = {
            'epochs': 5,
            'batch_size': 8,
        }

        init_kwargs = {
            'task': task,
            'timestamp': timestamp,
            'forecast_length': 3,
            'hidden_layer_units': 32,

            'y_scale': np.random.choice(['min_max', 'max_abs'], size=1)[0]
        }
        model = NBeatsWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score_rmse = rmse(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print('rmse:', score_rmse)
        print('mape:', score_mape)

        assert score_rmse >= 0
        assert score_mape >= 0


    def test_univariate_classification_inception(self):
        from hyperts.framework.wrappers.dl_wrappers import InceptionTimeWrapper

        X, y = load_arrow_head(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_UNIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,

            'blocks': 3,
            'learning_rate': 0.001,
            'reducelr_patience': 15,
            'earlystop_patience': 20,

            'x_scale': np.random.choice(['z_score', 'scale-none'], size=1)[0]
        }
        model = InceptionTimeWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_multivariate_classification_inception(self):
        from hyperts.framework.wrappers.dl_wrappers import InceptionTimeWrapper

        X, y = load_basic_motions(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.random_train_test_split(X, y, test_size=0.2)
        task = consts.Task_MULTIVARIATE_MULTICALSS

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 16,
        }

        init_kwargs = {
            'task': task,

            'blocks': 3,
            'learning_rate': 0.001,
            'reducelr_patience': 15,
            'earlystop_patience': 20,

            'x_scale': np.random.choice(['z_score', 'scale-none'], size=1)[0]
        }
        model = InceptionTimeWrapper(fit_kwargs, **init_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print('accuracy:', acc)

        assert acc >= 0


    def test_univariate_anomaly_detection(self):
        from hyperts.framework.wrappers.dl_wrappers import ConvVAEWrapper

        X, y = load_real_known_cause_dataset(return_X_y=True)
        tb = get_tool_box(X)
        X_train, X_test, y_train, y_test = tb.temporal_train_test_split(X, y, test_horizon=15000)

        fit_kwargs = {
            'epochs': 10,
            'batch_size': 64,
        }

        init_kwargs = {
            'task': 'detection',
            'timestamp': 'timestamp',
            'cnn_filters': 32,
            'contamination': 0.1,
            'x_scale': np.random.choice(['min_max', 'z_scale'], size=1)[0]
        }

        model = ConvVAEWrapper(fit_kwargs=fit_kwargs, **init_kwargs)

        model.fit(X_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        assert score >= 0