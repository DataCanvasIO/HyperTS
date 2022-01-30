import tensorflow
from tensorflow.python.keras import metrics
from tensorflow.python.util.tf_export import keras_export
from hyperts.framework.dl.losses import symmetric_mean_absolute_percentage_error

@keras_export('keras.metrics.SymmetricMeanAbsolutePercentageError')
class SymmetricMeanAbsolutePercentageError(metrics.MeanMetricWrapper):
    """Computes the symmetric mean absolute percentage error between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[SymmetricMeanAbsolutePercentageError()])
    ```
    """
    def __init__(self, name='symmetric_mean_absolute_percentage_error', **kwargs):
        super(SymmetricMeanAbsolutePercentageError, self).__init__(
            symmetric_mean_absolute_percentage_error, name=name, **kwargs)

metrics_custom_objects = {
    'SymmetricMeanAbsolutePercentageError': SymmetricMeanAbsolutePercentageError,
}

# tensorflow.keras.utils.get_custom_objects().update(metrics_custom_objects)