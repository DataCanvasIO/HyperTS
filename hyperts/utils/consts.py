# -*- coding:utf-8 -*-


TIMESTAMP                          = 'timestamp'
DEFAULT_EVAL_SIZE                  = 0.2

Task_UNIVARIABLE_FORECAST          = 'univariable-forecast'
Task_MULTIVARIABLE_FORECAST        = 'multivariable-forecast'
Task_UNIVARIABLE_BINARYCLASS       = 'univariable-binaryclass'
Task_MULTIVARIABLE_BINARYCLASS     = 'multivariable-binaryclass'
Task_UNIVARIABLE_MULTICALSS        = 'univariable-multiclass'
Task_MULTIVARIABLE_MULTICALSS      = 'multivariable-multiclass'
Task_FORECAST                      = 'forecast'
Task_CLASSIFICATION                = 'classification'
Task_REGRESSION                    = 'regression'

TASK_LIST_FORECAST                 = ['forecast',
'univariable-forecast', 'multivariable-forecast'
]

TASK_LIST_CLASSIFICATION           = ['classification',
'univariable-binaryclass', 'multivariable-binaryclass',
'univariable-multiclass', 'multivariable-multiclass'
]

TASK_LIST_REGRESSION               = ['regression'
]

Mode_STATS                         = 'stats'
Mode_DL                            = 'dl'
Mode_NAS                           = 'nas'

DataType_INT                       = 'int'
DataType_FLOAT                     = 'float'
DataType_OBJECT                    = 'object'

StepName_DATA_PREPROCESSING        = 'data_preprocessing'
StepName_SPACE_SEARCHING           = 'space_searching'
StepName_FINAL_TRAINING            = 'final_training'

OptimizeDirection_MINIMIZE         = 'min'
OptimizeDirection_MAXIMIZE         = 'max'

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