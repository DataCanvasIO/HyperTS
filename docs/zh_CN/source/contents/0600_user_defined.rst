自定义化
########
HyperTS除了使用内置的算法外, 还支持用户自定义部分功能, 以增强其扩展性。


自定义评估指标
==============

当使用 ``make_experiment`` 创建实验时, 您可以通过参数 ``reward_metric`` 重新指定评估指标, 示例如下:

.. code-block:: python

    from hyperts import make_experiment

    experiment = make_experiment(train_data, 
                                task='forecast',
                                timestamp='TimeStamp',
                                reward_metric='mae',
                                ...) 

除了传入内置支持的评估指标, 您也可以自定义评估指标来满足特定场景下的需求, 例如:

方式一:

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

    当采用这种方式自定评估指标时, 需指定优化方向optimize_direction。

方式二:

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

    当采用这种方式自定评估指标时, 需设置参数scorer。

----------------

自定义搜索空间
==============

HyperTS针对不同的模式内置了丰富的建模算法, 例如:

- StatsForecastSearchSpace: 预测任务统计模型搜索空间, 内置了Prophet、ARIMA及VAR等统计模型;
- StatsClassificationSearchSpace: 分类任务统计模型搜索空间, 内置了TSForest, k-NNs等统计模型;
- DLForecastSearchSpace: 预测任务深度模型搜索空间, 内置DeepAR、RNN、GPU、LSTM及LSTNet等深度模型;
- DLClassificationSearchSpace: 分类任务深度模型搜索空间, 内置RNN、GPU、LSTM及LSTNet等深度模型。
  
以上建模算法均设计了各自默认的超参数搜索空间。如果您想在此基础上定制化自己的搜索空间, 则可以在调用 ``make_experiment`` 时通过参数 ``search_space`` 指定自定义的搜索空间。

假如现在我们想修改预测任务下的统计模式的搜索空间, 即 ``StatsForecastSearchSpace``, 您可以做如下操作:

- 如果想禁止某个算法, 不进行搜索, 可以设置参数为False, 例如 ``enable_arima=False``;
- 如果想更改某个算法的搜索空间参数初始化,可以传递参数 ``xxx_init_kwargs={xxx:xxx, ...}``;
- 如果希望自定义的参数是可搜索的, 您可以使用 ``hypernets.core.search_space`` 中的 ``Choice``, ``Int`` 及 ``Real``。其中, ``Choice`` 支持离散值, ``Int`` 支持整数连续值, ``Real`` 支持浮点数连续值。详情可参考 `Search Space <https://github.com/DataCanvasIO/Hypernets/blob/master/hypernets/core/search_space.py>`_。

代码示例:

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

--------------

自定义建模算法
==============

在自定义搜索空间中, 我们提到HyperTS针对不同的模式内置了丰富的建模算法, 如果您需要增加对其他算法的支持, 可以通过如下步骤进行自定义建模算法:

- 将自定义算法封装为 ``HyperEstimator`` 的子类;
- 将封装后的算法增加到特定任务的SearchSpace中, 并定义其搜索参数;
- 在 ``make_experiment`` 中使用自定义的search_space。

假如现在我们想在 **DLForecastSearchSpace** 中增加自己的神经网络模型 **Transformer**, 示例如下:

构建自定义模型
**************

我们基于tensorflow构建一个 *Transformer Encoder*, 该结构参看自 `Keras官方教程 <https://keras.io/examples/timeseries/timeseries_classification_transformer/>`_。

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
 
构建自定义算法
**************

为了方便起见, 我们可以直接继承HyperTS中已存在的算法, 这样除了必要的init部分外, 我们只完成 ``_build_estimator`` 中的backbone部分即可, 即:

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

构建算法估计器
**************

估计器将是连接算法模型与搜索空间的桥梁与纽带, 它可以规定哪些超参数将能够被搜索优化。

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

重构搜索空间
************

将估计器添加到搜索空间中就大功告成啦! 在这里, 我们可以设置自定义算法中一些超参数的搜索空间, 这一步将是发挥算法性能的关键:

.. code-block:: python

    from hypernets.core.search_space import Choice, Real
    from hyperts.framework.search_space.macro_search_space import DLForecastSearchSpace


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


使用新搜索空间执行实验
**********************

这里将和前面介绍的自定义搜索空间的操作一致。

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
