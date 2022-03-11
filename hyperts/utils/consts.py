# -*- coding:utf-8 -*-
from hypernets.utils.const import *

TIMESTAMP                          = 'timestamp'
DEFAULT_EVAL_SIZE                  = 0.2
DEFAULT_MIN_EVAL_SIZE              = 0.05
NAN_DROP_SIZE                      = 0.6
FINAL_TRAINING_EPOCHS              = 120

Task_UNIVARIATE_FORECAST           = 'univariate-forecast'
Task_MULTIVARIATE_FORECAST         = 'multivariate-forecast'
Task_UNIVARIATE_BINARYCLASS        = 'univariate-binaryclass'
Task_MULTIVARIATE_BINARYCLASS      = 'multivariate-binaryclass'
Task_UNIVARIATE_MULTICALSS         = 'univariate-multiclass'
Task_MULTIVARIATE_MULTICALSS       = 'multivariate-multiclass'
Task_FORECAST                      = 'forecast'
Task_CLASSIFICATION                = 'classification'
Task_REGRESSION                    = 'regression'

TASK_LIST_FORECAST                 = ['forecast',
'univariate-forecast', 'multivariate-forecast'
]

TASK_LIST_CLASSIFICATION           = ['classification',
'univariate-binaryclass', 'multivariate-binaryclass',
'univariate-multiclass', 'multivariate-multiclass'
]

TASK_LIST_REGRESSION               = ['regression'
]

TASK_LIST_BINARYCLASS              = ['univariate-binaryclass', 'multivariate-binaryclass']

TASK_LIST_MULTICLASS               = ['univariate-multiclass', 'multivariate-multiclass']

Mode_STATS                         = 'stats'
Mode_DL                            = 'dl'
Mode_NAS                           = 'nas'

DataType_INT                       = 'int'
DataType_FLOAT                     = 'float'
DataType_OBJECT                    = 'object'
DATATYPE_TENSOR_FLOAT              = 'float32'

StepName_DATA_PREPROCESSING        = 'data_preprocessing'
StepName_SPACE_SEARCHING           = 'space_searching'
StepName_FINAL_TRAINING            = 'final_training'
StepName_FINAL_ENSEMBLE            = 'final_ensemble'

OptimizeDirection_MINIMIZE         = 'min'
OptimizeDirection_MAXIMIZE         = 'max'

Searcher_RONDOM                    = 'random'
Searcher_EVOLUTION                 = 'evolution'
Searcher_MCTS                      = 'mcts'

Metric_MSE                         = 'mse'
Metric_RMSE                        = 'rmse'
Metric_MAE                         = 'mae'
Metric_MAPE                        = 'mape'
Metric_SMAPE                       = 'smape'
Metric_R2                          = 'r2'
Metric_MSLE                        = 'msle'
Metric_ACCURACY                    = 'accuracy'
Metric_PRESICION                   = 'precision'
Metric_RECALL                      = 'recall'
Metric_AUC                         = 'auc'
Metric_F1                          = 'f1'
Metric_LOGLOSS                     = 'logloss'

OptimizerSGD                       = 'sgd'
OptimizerADAM                      = 'adam'