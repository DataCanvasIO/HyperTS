Advanced Configurations
########################

The :doc:`Quick Start </contents/0400_quick_start>` section presents the most basic application of HyperTS. The example is repeated as below. 

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



Default Settings
===============================

Firstly, load the input data and define the ``task`` type. The dataset information are collected `here <https://github.com/DataCanvasIO/HyperTS/blob/main/hyperts/datasets/base.py>`_。
Secondly, define the related variables according to different task types. For forecasting task, the required variables are the ``timestampe``, and ``covariables`` if have. For classification task, the required variable is the ``target``. 

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
  
  If the exact task type is known, the specific name is recommended to assign to the function argument. For example, ``task='univariate-forecast'``. If not, using the general type name, ``task='classification'``, also works. 



Select the Run Mode
=============================

HyperTS includes three types of methods: Statistical (default), Deep Learning and Neural Architecture Search, which are abbreviated as ``stats``, ``dl`` and ``nas`` respectively. Users could select the methods by setting argument ``mode``. 

.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              ...)                            

The deep learning method is based on the Tensorfolw framework, which processes in CPU by default and also supports GPU after installing tensorflow-gpu. There are in total three usage strategies: 

- 0: processing in CPU;
- 1: processing in GPU with increasing memory according to the data scale;  
- 2: processing in GPU with limited memory (2048M). Change the memory limit by the argument ``dl_memory_limit``.


.. code-block:: python

  experiment = make_experiment(train_data, 
                              mode='dl',
                              dl_gpu_usage_strategy=1,
                              ...)                            



Set the Evaluation Metric
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

- Classification: accuracy, auc, f1, precision, recall, logloss。
- Forecasting and regression: mae, mse, rmse, mape, smape, msle, r2。



Set the Optimization Direction
================================

The searcher needs an indication of the optimization direction ('min' or 'max'). By default, the system will detect from ``reward_metric``.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='univariate-binaryclass',
                              reward_metric='auc',
                              optimize_direction='max',
                              ...)                            



Set the Max Search Trials
============================

The default search trials is only three to obtain quick results. In practice, to achieve better performace, the search trails value is recommended more than 30. The higher the ``max_trials`` value is, the better performace would obtain if the time is sufficient.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              ...)                     



Set the Early Stopping Strategy
===============================

The early stopping strategy could define three different criterions to stop the processing to save time. The three strategies are:

- ``early_stopping_time_limit``:  unit is second.
- ``early_stopping_round``: the trials number of invalid search after obtaining the optimal value.
- ``early_stopping_reward``: defines the threshold value of certain reward.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              max_trials=100,
                              early_stopping_time_limit=3600 * 3,  # set the max search time is three hours
                              ...)    
                        


Define the Evaluation Dataset
==============================

The evaluation dataset is split from the training dataset by default. Users could adjust ``eval_size`` to set the percentage. 

.. code-block:: python

  experiment = make_experiment(train_data, 
                              eval_size=0.3,
                              ...)                           

Besides, users could define a certain dataset as evaluation dataset by setting the argument ``eval_data``. 

.. code-block:: python

  experiment = make_experiment(train_data, 
                              eval_data=eval_data,
                              ...)                            



Define a Searcher
======================

HyperTS performs the model selection and hyperparameter search by the built-in search algorithms in `Hypernets <https://github.com/DataCanvasIO/Hypernets>`_, which includes EvolutionSearch(default, 'evalution'), MCTSSearcher('mcts'), RandomSearcher('random') and GridSearch('grid'). Users could define a specific search by setting the argument ``searcher``. It could be a class name or a string of the name.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              searcher='random',
                              ...)                            

For more details of the search algorithms, please refer to the section `Search Algorithm <https://hypernets.readthedocs.io/en/latest/searchers.html>`_.



Set the Time Frequency
=======================

For time series forecasting task, users could set the desired time frequency by the argument ``freq``. The provided options are second (`S`), minute('T')、hour('H')、day('D')、week('W')、month('M') and year('Y'). If the frequency information is missing, it will adjust according to ``timestamp``.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='forecast',
                              timestamp='TimeStamp',
                              freq='H',
                              ...) 




Set the Time Window
=============================

When selecting the deep learning mode, users could set argument ``dl_forecast_window`` to define the size of moving time window. The unit is per hour.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              task='forecast',
                              mode='dl',
                              timestamp='TimeStamp',
                              dl_forecast_window=24*7,
                              ...)                            



Fix the Random Seed
==========================

Sometimes, the codes need to be re-executed. In order to keep the random numbers fixed, users could set the argument ``random_state``. 

.. code-block:: python

  experiment = make_experiment(train_data, 
                              random_state=0,
                              ...)                            



Set the Log Level
=======================

The progress messages during training can be printed by the argument ``log_level``. For more information, please refer to the python package ``logging``. Besides, more comprehensive messages will be printed when setting ``verbose = 1``.

.. code-block:: python

  experiment = make_experiment(train_data, 
                              log_level='INFO', 
                              verbose=1,
                              ...)                            
