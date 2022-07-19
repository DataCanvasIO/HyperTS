User-defined Functions
#######################

HyperTS supports several user-defined extension functions in addition to the built-in algorithms. 


User-defined Evaluation Metric
================================

When creating an experiment, the evaluation criterion could be set by the argument ``reward_metric``. See example below:

.. code-block:: python

    from hyperts import make_experiment

    experiment = make_experiment(train_data, 
                                task='forecast',
                                timestamp='TimeStamp',
                                reward_metric='mae',
                                ...) 

Except these built-in criterions, users could define their own criterion. Two approaches are introduced below: 

Approach one:

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error

    def custom_metric(y_true, y_pred, epsihon=1e-06):
        if (y_true < 0).any():
            y_true = np.clip(y_true, a_min=epsihon, a_max=abs(y_true))

        if (y_pred < 0).any():
            y_pred = np.clip(y_pred, a_min=epsihon, a_max=abs(y_pred))

        return mean_squared_error(np.log1p(y_true), np.log1p(y_pred))


    experiment = make_experiment(train_data, 
                                task='forecast',
                                timestamp='TimeStamp',
                                reward_metric=custom_metric,
                                optimize_direction='min',
                                ...) 

.. note::

    In this case, the ``optimize_direction`` is required.

Approach two:

.. code-block:: python

    import numpy as np
    from sklearn.metrics import mean_squared_error, make_scorer

    def custom_metric(y_true, y_pred):
        if (y_true < 0).any():
            y_true = np.clip(y_true, a_min=epsihon, a_max=abs(y_true))

        if (y_pred < 0).any():
            y_pred = np.clip(y_pred, a_min=epsihon, a_max=abs(y_pred))

        return mean_squared_error(np.log1p(y_true), np.log1p(y_pred))

    custom_scorer = make_scorer(custom_metric, greater_is_better=True, needs_proba=False)

    experiment = make_experiment(train_data, 
                                task='forecast',
                                timestamp='TimeStamp',
                                reward_metric=custom_metric,
                                scorer=custom_scorer,
                                ...) 

.. note::

    In this case, the ``scorer`` is required. 


User-defined Search Space
=========================

HyperTS provides various algorithms with default search space for every mode. Most of them are listed below:

- 'StatsForecastSearchSpace' includes the search space for statistical models (Prophet, ARIMA, VAR);
- 'StatsClassificationSearchSpace' includes the search space for statistical models (TSForest, k-NNs);
- 'DLForecastSearchSpace' includes the search space for deep learning models (DeepAR, RNN, GPU, LSTM, LSNet);
- 'DLClassificationSearchSpace' includes the search space for deep learning models (RNN, GPU, LSTM, LSNet).
  
By setting the argument ``search_space``, users could define their own search space. The instructions and an example are given below to modify the ``StatsForecastSearchSpace``. 

- Set the argument as false to disable a certain algorithm. For instance, ``enable_arima=False``;
- Change the initial parameters of a certain algorithm by function ``prophet_init_kwargs={xxx:xxx, ...}``;
- Import the argument ``Choice``, ``Int`` ``Real`` from ``hypernets.core.search_space`` could define the parameters with specific options. For instance, ``Choice`` supports the boolean data type. ``Real`` supports the floating data type.
- For more information, please refer to `Search Space <https://github.com/DataCanvasIO/Hypernets/blob/master/hypernets/core/search_space.py>`_.

Code example:

.. code-block:: python

    from hypernets.core.search_space import Choice, Int, Real
    from hyperts.framework.search_space.macro_search_space import StatsForecastSearchSpace

    custom_search_space = StatsForecastSearchSpace(enable_arima=False,
                                                   prophet_init_kwargs={
                                                    'seasonality_mode': 'multiplicative',
                                                    'daily_seasonality': Choice([True, False]),
                                                    'n_changepoints': Int(10, 50, step=10),
                                                    'interval_width': Real(0.1, 0.5, step=0.1)}
                                                )

    experiment = make_experiment(train_data, 
                                task='univariate-forecast',
                                timestamp='TimeStamp',
                                covariables=['HourSin', 'WeekCos', 'CBWD'],
                                search_space=custom_search_space,
                                ...) 



User Defined Modeling Algorithm
================================

In addition to the built-in modeling algorithms as mentioned above, users could also define new algorithms. The instructions and an example are given below to build a modified neural network model 'Transformer' inside 'DLForecastSearchSpace':

- Package the user-modified algorithm as a subclass of ``HyperEstimator``;
- Add the subclass to the specific search space and define the search parameters; 
- Assign the search space to the argument of function ``make_experiment``.

Code example

1. Build the Model Structure
*****************************

The example is to build a *Transformer Encoder* based on tensorflow. See `Keras tutorial <https://keras.io/examples/timeseries/timeseries_classification_transformer/>`_.

.. code-block:: python

    from tensorflow.keras import layers

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
 
2. Build the Algorithm
***********************

To make it sample, this example uses a template of an existing algorithm in HyperTS. Only a small part of ``_init_`` and ``_build_estimator`` are modified. 

.. code-block:: python

    import tensorflow as tf
    import tensorflow.keras.backend as K
    from hyperts.framework.dl import layers
    from hyperts.framework.dl.models import HybirdRNN

    class Transformer(HybirdRNN):

        def __init__(self, 
                    task, 
                    timestamp=None, 
                    window=7, 
                    horizon=1, 
                    forecast_length=1, 
                    head_size=10,
                    num_heads=6,
                    ff_dim=10,
                    transformer_blocks=1,
                    drop_rate=0.,
                    metrics='auto',
                    monitor_metric='val_loss',
                    optimizer='auto',
                    learning_rate=0.001,
                    loss='auto',
                    out_activation='linear',
                    reducelr_patience=5, 
                    earlystop_patience=10, 
                    embedding_output_dim=4,
                    **kwargs):
            super(Transformer, self).__init__(task=task, 
                                            timestamp=timestamp, 
                                            window=window, 
                                            horizon=horizon, 
                                            forecast_length=forecast_length,
                                            drop_rate=drop_rate,
                                            metrics=metrics, 
                                            monitor_metric=monitor_metric, 
                                            optimizer=optimizer,
                                            learning_rate=learning_rate, 
                                            loss=loss, 
                                            out_activation=out_activation, 
                                            reducelr_patience=reducelr_patience, 
                                            earlystop_patience=earlystop_patience,
                                            embedding_output_dim=embedding_output_dim, 
                                            **kwargs)
            self.head_size = head_size
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.transformer_blocks = transformer_blocks

        
        def _build_estimator(self, **kwargs):
            K.clear_session()
            continuous_inputs, categorical_inputs = layers.build_input_head(self.window, self.continuous_columns, self.categorical_columns)
            denses = layers.build_denses(self.continuous_columns, continuous_inputs)
            embeddings = layers.build_embeddings(self.categorical_columns, categorical_inputs)
            if embeddings is not None:
                x = layers.Concatenate(axis=-1, name='concat_embeddings_dense_inputs')([denses, embeddings])
            else:
                x = denses  

            ############################################ backbone ############################################
            for _ in range(self.transformer_blocks):
                x = transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.drop_rate)
            x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            ##################################################################################################

            outputs = layers.build_output_tail(x, self.task, nb_outputs=self.meta.classes_, nb_steps=self.forecast_length)
            outputs = layers.Activation(self.out_activation, name=f'output_activation_{self.out_activation}')(outputs)

            all_inputs = list(continuous_inputs.values()) + list(categorical_inputs.values())
            model = tf.keras.models.Model(inputs=all_inputs, outputs=[outputs], name=f'Transformer')
            model.summary()
            return model

1. Build the Estimator
***********************

Estimator connectes the algorithm model and search space. It defines the hyperparameters for optimization.

.. code-block:: python

    from hyperts.utils import consts
    from hyperts.framework.wrappers.dl_wrappers import HybirdRNNWrapper
    from hyperts.framework.estimators import HyperEstimator

    class TransformerWrapper(HybirdRNNWrapper):

        def __init__(self, fit_kwargs, **kwargs):
            super(TransformerWrapper, self).__init__(fit_kwargs, **kwargs)
            self.update_dl_kwargs()
            self.model = Transformer(**self.init_kwargs)


    class TransfomerEstimator(HyperEstimator):

        def __init__(self, fit_kwargs=None, timestamp=None, task='univariate-forecast', window=7,
                    head_size=10, num_heads=6, ff_dim=10, transformer_blocks=1, drop_rate=0.,
                    metrics='auto', optimizer='auto', out_activation='linear',
                    learning_rate=0.001, batch_size=None, epochs=1, verbose=1,
                    space=None, name=None, **kwargs):

            if task in consts.TASK_LIST_FORECAST and timestamp is None:
                raise ValueError('Timestamp need to be given for forecast task.')
            else:
                kwargs['timestamp'] = timestamp
            if task is not None:
                kwargs['task'] = task
            if window is not None and window != 7:
                kwargs['window'] = window
            if head_size is not None and head_size != 10:
                kwargs['head_size'] = head_size
            if num_heads is not None and num_heads != 6:
                kwargs['num_heads'] = num_heads
            if ff_dim is not None and ff_dim != 10:
                kwargs['ff_dim'] = ff_dim
            if transformer_blocks is not None and transformer_blocks != 1:
                kwargs['transformer_blocks'] = transformer_blocks
            if drop_rate is not None and drop_rate != 0.:
                kwargs['drop_rate'] = drop_rate
            if metrics is not None and metrics != 'auto':
                kwargs['metrics'] = metrics
            if optimizer is not None and optimizer != 'auto':
                kwargs['optimizer'] = optimizer
            if out_activation is not None and out_activation != 'linear':
                kwargs['out_activation'] = out_activation
            if learning_rate is not None and learning_rate != 0.001:
                kwargs['learning_rate'] = learning_rate 

            if batch_size is not None:
                    kwargs['batch_size'] = batch_size
            if epochs is not None and epochs != 1:
                kwargs['epochs'] = epochs
            if verbose is not None and verbose != 1:
                kwargs['verbose'] = verbose

            HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

        def _build_estimator(self, task, fit_kwargs, kwargs):
            if task in consts.TASK_LIST_FORECAST + consts.TASK_LIST_CLASSIFICATION:
                transformer = TransformerWrapper(fit_kwargs, **kwargs)
            else:
                raise ValueError('Check whether the task type meets specifications.')
            return transformer

4.  Build the Search Space
***************************

Add the estimator to the search space, in which the hyperparameters also could be defined properly to ensure the performance.  

.. code-block:: python

    from hypernets.core.search_space import Choice, Real
    from hyperts.framework.macro_search_space import DLForecastSearchSpace


    class DLForecastSearchSpacePlusTransformer(DLForecastSearchSpace):

        def __init__(self, task, timestamp=None, metrics=None, window=None, enable_transformer=True, **kwargs):
            super().__init__(task=task, timestamp=timestamp, metrics=metrics, window=window, **kwargs)
            self.enable_transformer = enable_transformer

        @property
        def default_transformer_init_kwargs(self):
            return {
                'timestamp': self.timestamp,
                'task': self.task,
                'metrics': self.metrics,

                'head_size': Choice([8, 16, 24, 32]),
                'num_heads': Choice([2, 4, 6]),
                'ff_dim': Choice([8, 16, 24, 32]),
                'drop_rate': Real(0., 0.5, 0.1),
                'transformer_blocks': Choice([1, 2, 3]),            
                'window': self.window if self.window is not None else Choice([12, 24, 48]),

                'y_log': Choice(['logx', 'log-none']),
                'y_scale': Choice(['min_max', 'max_abs'])
            }

        @property
        def default_transformer_fit_kwargs(self):
            return {
                'epochs': 60,
                'batch_size': None,
                'verbose': 1,
            }

        @property
        def estimators(self):
            r = super().estimators
            if self.enable_transformer:
                r['transformer'] = (TransfomerEstimator, self.default_transformer_init_kwargs, self.default_transformer_fit_kwargs)
            return r


5. Execute the Experiment with Custom Search Space
***************************************************

.. code-block:: python

    from hyperts import make_experiment
    from hyperts.datasets import load_network_traffic
    from sklearn.model_selection import train_test_split

    df = load_network_traffic(univariate=True)
    train_data, test_data = train_test_split(df, test_size=168, shuffle=False)

    custom_search_space = DLForecastSearchSpacePlusTransformer()

    experiment = make_experiment(train_data, 
                                task='univariate-forecast',
                                mode='dl',
                                timestamp='TimeStamp',
                                covariables=['HourSin', 'WeekCos', 'CBWD'],
                                search_space=custom_search_space,
                                reward_metric='mape',
                                ...)

    model = experiment.run() 
