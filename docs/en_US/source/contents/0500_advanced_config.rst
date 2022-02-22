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



Select the processing method
==================

HyperTS includes three types of processing methods: Statistical (default)， Deep Learning and Neural Architecture Search, which are abbreviated as ``stats``, ``dl`` and ``nas`` respectively. Users could select the methods by setting augument ``mode``.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              ...)                            

The deep learning method is based on the Tensorfolw framework, which processes in CPU by default and also supports GPU after installing tensorflow-gpu. There are three usage stratrges: 
- 0: processing in CPU;
- 1: processing in GPU with increasing memory according to the data scale;  
- 2: processing in GPU with limited memory (2048M). Change the memory limit by the argument ``dl_memory_limit``.


.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              dl_gpu_usage_strategy=1,
                              ...)                            



Set the evaluation criterion
=================================

By default, the evaluation criterion for forecasting task is 'mae', for classification task 'accuracy' and for regression task 'rmse'. Users could also set other evaluation criterion by argument ``reward-metric`` in both string format or importing from ``sklearn.metrics``.

.. code-block:: python

  # string format
  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric='auc',
                              ...)  

  # from sklearn.metrics
  from sklearn.metrics import auc
  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric= auc,
                              ...)                                                        

Currently, ``reward_metric`` supports the following criterion: 

- classification: accuracy, auc, f1, precision, recall, logloss。
- forecasting and regression: mae, mse, rmse, mape, smape, msle, r2。



Define the optimization direction
================================

The searcher needs an indication of the optimization direction ('min' or 'max'). By default, the system will detect from ``reward_metric``.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric='auc',
                              optimize_direction='max',
                              ...)                            

------------------

Set the max search trials value
============================

The default search trial is three to obtain quick results. In practice, to achieve better performace, the search trails value is recommended more than 30. The higher the ``max_trials`` value is, the better performace would obtain if the time is sufficient.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              ...)                     



Set the early stopping strategy
============================

The early stopping strategy could define three different criterions to stop the processing to save time. The three strategies are:
- ``early_stopping_time_limit``:  unit is second.
- ``early_stopping_round``: limit is the times of search trials (priority to max_trials).
- ``early_stopping_reward``: defines the threshold value of certain reward.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              early_stopping_time_limit=3600 * 3,  # set the max search time is three hours
                              ...)    
                        


Define the evaluation dataset
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
