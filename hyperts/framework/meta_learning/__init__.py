from .helper_fn import ptp, normalization
from .tsfeatures import metafeatures_from_timeseries


metric_mapping_dict = {
    'auc': 'roc_auc_score',
    'roc_auc_score': 'roc_auc_score',
    'accuracy': 'accuracy',
    'accuracy_score': 'accuracy',
    'recall': 'recall',
    'recall_score': 'recall',
    'precision': 'precision',
    'precision_score': 'precision',
    'f1': 'f1_score',
    'f1_score': 'f1_score',
    'r2': 'r2_score',
    'r2_score': 'r2_score',
    'logloss': 'neg_log_loss',
    'log_loss': 'neg_log_loss',

    'mse': 'mean_squared_error',
    'mean_squared_error': 'mean_squared_error',
    'neg_mean_squared_error': 'mean_squared_error',
    'mae': 'mean_absolute_error',
    'mean_absolute_error': 'mean_absolute_error',
    'neg_mean_absolute_error': 'mean_absolute_error',
    'msle': 'mean_squared_log_error',
    'mean_squared_log_error': 'mean_squared_log_error',
    'neg_mean_squared_log_error': 'mean_squared_log_error',
    'rmse': 'root_mean_squared_error',
    'root_mean_squared_error': 'root_mean_squared_error',
    'neg_root_mean_squared_error': 'root_mean_squared_error',
    'mape': 'mean_absolute_percentage_error',
    'mean_absolute_percentage_error': 'mean_absolute_percentage_error',
    'neg_mean_absolute_percentage_error': 'mean_absolute_percentage_error',
    'smape': 'symmetric_mean_absolute_percentage_error',
    'symmetric_mean_absolute_percentage_error': 'symmetric_mean_absolute_percentage_error',
    'neg_symmetric_mean_absolute_percentage_error': 'symmetric_mean_absolute_percentage_error'
}