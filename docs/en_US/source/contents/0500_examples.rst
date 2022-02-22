Advanced Configurations
########

:doc:`Quick Start </contents/0400_quick_start>` presents the most basic application of HyperTS. It's repeated as below. 

.. code-block:: python

  from hyperts.experiment import make_experiment
  from hyperts.datasets import load_network_traffic
  from sklearn.model_selection import train_test_split

  df = load_network_traffic()
  train_data, test_data = train_test_split(df, test_size=168, shuffle=False)

  experiment = make_experiment(train_data, 
                              task='forecast',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'])
  model = experiment.run()

  X_test, y_test = model.split_X_y(test_data)
  forecast = model.predict(X_test)
  scores = model.evaluate(y_true=y_test, y_pred=forecast)
  ...

This section will introduce some advanced configurations of ``make_experience`` to help to achieve more robust and better performance. 



Default settings
===============================

Firstly, load the input data and define the ``task`` type. The related data information is `here <https://github.com/DataCanvasIO/HyperTS/blob/main/hyperts/datasets/base.py>`_。
Secondly, define the related variables according to different task types. For forecasting task, the required variables are the ``timestampe``, and ``covariables`` if have. For classification task, the required avariable is the ``target``. 

Example codes:

.. code-block:: python

  #Forecasting task
  experiment = make_experiment(train_data, 
                              task='forecast',
                              timestamp='TimeStamp',
                              covariables=['HourSin', 'WeekCos', 'CBWD'])
  #Classification task
  experiment = make_experiment(train_data, task='classification', target='y')                            


.. note::

  The time series forecasting task could be further divided into ``univariate-forecast`` and ``multivariate-forecast`` depending on the number of the forecast variables.
  
  For claasification task, it could be divided into ``univariate-binaryclass``, ``univariate-multiclass``, ``multivariate-binaryclass`` and ``multivariate-multiclass``, according to the number of features and the target categories. 
  
  If the exact task type is known, the specific name is recommended to assign to the function argument. For example, ``task='univariate-forecast'``. If not, using the general type name, ``task='classification'``, also work. 



选择运行模式(mode)
==================

HyperTS内置了三种运行模式, 分别为统计模型模式('stats'), 深度学习模式('dl')以及神经架构搜索模式('nas', 未开放)。缺省情况下, 默认选择统计模型模式, 您也可以更改为其他模式:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              ...)                            

深度学习模式基于Tensorflow可支持GPU, 缺省情况下, 默认实验将在CPU环境下运行。如果您的设备支持GPU并安装了gpu版本的tensorflow-gpu, 可以更改参数 ``dl_gpu_usage_strategy```:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              dl_gpu_usage_strategy=1,
                              ...)                            

其中, ``dl_gpu_usage_strategy`` 支持三种配置策略, 分别为:

- 0: CPU下运行;
- 1: GPU内存容量依据模型规模及运行情况增长;
- 2: GPU内存容量限制最大容量, 默认为2048M, 参数 ``dl_memory_limit`` 支持自定义配置。

------------------

指定模型的评估指标(reward_metric)
=================================

当使用 ``make_experiment`` 创建实验时, 缺省情况下, 预测任务默认的模型评估指标是'mae', 分类任务是'accuracy', 回归任务默认是'rmse'。您可以通过参数 ``reward_metric`` 重新指定评估指标, 可以是'str'也可以是 ``sklearn.metrics`` 内置函数, 示例如下:

.. code-block:: python

  # str
  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric='auc',
                              ...)  

  # sklearn.metrics
  from sklearn.metrics import auc
  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric=auc,
                              ...)                                                        

目前, ``reward_metric`` 可以支持多种评估指标, 具体如下: 

- 分类: accuracy, auc, f1, precision, recall, logloss。
- 预测及回归: mae, mse, rmse, mape, smape, msle, r2。

------------------

指定优化方向(optimize_direction)
================================

在模型搜索阶段, 需要给搜索者指定搜索方向, 在缺省情况下, 默认将从 ``reward_metric`` 中检测。您也可以通过参数 ``optimize_direction`` 进行指定('min'或者'max'):

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric='auc',
                              optimize_direction='max',
                              ...)                            

------------------

设置最大搜索次数(max_trials)
============================

缺省情况下, ``make_experiment`` 所创建的实验搜索3种参数模型便停止搜索。实际使用中, 建议将最大搜索次数设置为30以上, 时间充裕的话, 更大的搜索次数将有更高的机率获得更加优秀的模型:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              ...)                     

------------------

设置早停策略(early_stopping)
============================

当 ``max_trials`` 设置比较大时, 可能需要更多的时间等待实验运行完毕。为了把控工作的节奏, 您可以通过 ``make_experiment`` 的早停机制(Early Stopping)进行控制:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              early_stopping_time_limit=3600 * 3,  # 将搜索时间设置为最多3个小时
                              ...)    
                        
其中, ``make_experiment`` 共包含了三种早停机制, 分别为:

- early_stopping_time_limit: 限制实验的运行时间, 粒度为秒。
- early_stopping_round: 限制实验的搜索轮数, 粒度为次。
- early_stopping_reward: 指定一个奖励得分的界限。

------------------

指定验证数据集(eval_data)
=========================

模型训练除了需要训练数据集, 还需要评估数据集, 缺省情况下将从训练数据集中以一定比例切分一部分评估数据集。您也可在 ``make_experiment`` 时通过eval_data指定评估集, 如:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              eval_data=eval_data,
                              ...)                           

当然, 您也可以通过设置 ``eval_size`` 自己指定评估数据集的大小:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              eval_size=0.3,
                              ...)                            

------------------

指定搜索算法(searcher)
======================

HyperTS通过 `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_ 中内置的搜索算法进行模型选择和超参数优化, 其中包括EvolutionSearcher(缺省, 'evolution')、MCTSSearcher('mcts')、RandomSearch('random')以及GridSearch('grid')等。在使用 ``make_experiment`` 时, 可通过参数 ``searcher`` 指定, 指定搜索算法的类名(class)或者搜索算法的名称(str):

.. code-block:: python

  experiment = make_experiment(train_data, 
                              searcher='random',
                              ...)                            

各种搜索算法详细介绍可参考 `搜索算法 <https://hypernets.readthedocs.io/en/latest/searchers.html>`_。

------------------

指定时间频率(freq)
==================

在时序预测任务中, 如果我们已知数据集的时间频率, 您可以通过参数 ``freq`` 来精确化指定:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='forecast',
                              timestamp='TimeStamp',
                              freq='H',
                              ...) 

缺省情况下, 频率将依据 ``timestamp`` 进行推断。                              

------------------

指定预测窗口(forecast_window)
=============================

当使用深度学习模式进行时序预测时, 您可以结合经验对数据的实际情况分析后, 通过参数 ``forecast_window`` 指定滑动窗口的大小:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='forecast',
                              mode='dl',
                              timestamp='TimeStamp',
                              forecast_window=24*7,
                              ...)                            

------------------

固定随机种子(random_state)
==========================

有时为了保证实验结果可以复现, 我们需要保持相同的初始化, 此时, 您可以通过参数 ``random_state`` 固定随机种子:

.. code-block:: python

  experiment = make_experiment(train_data, 
                              random_state=0,
                              ...)                            

------------------

调整日志级别(log_level)
=======================

如果希望在训练过程中看到使用进度信息的话, 可通过log_level指定日志级别。关于日志级别的详细定义可参考python的logging包。 另外, 如果将verbose设置为1的话, 可以得到更详细的信息。例如, 将日志级别设置为'INFO':

.. code-block:: python

  experiment = make_experiment(train_data, 
                              log_level='INFO', 
                              verbose=1,
                              ...)                            
