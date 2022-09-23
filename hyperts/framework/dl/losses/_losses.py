import math
import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.losses.LogGaussianLoss')
class LogGaussianLoss(losses.LossFunctionWrapper):
    """Log Gaussian loss, is applied to DeepAR.

    Args:
        name: (Optional) string name of the metric instance.

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss=LogGaussianLoss(),
      metrics=['mse'])
    ```
    """
    def __init__(self, name='log_gaussian_loss', **kwargs):
        super(LogGaussianLoss, self).__init__(log_gaussian_error, name=name, **kwargs)


@keras_export('keras.losses.SymmetricMeanAbsolutePercentageError')
class SymmetricMeanAbsolutePercentageError(losses.LossFunctionWrapper):
    """Symmetric Mean Absolute Percentage Error loss.

    Args:
        name: (Optional) string name of the metric instance.

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss=SymmetricMeanAbsolutePercentageLoss(),
      metrics=['mse'])
    ```
    """
    def __init__(self, name='symmetric_mean_absolute_percentage_loss', **kwargs):
        super(SymmetricMeanAbsolutePercentageError, self).__init__(
            symmetric_mean_absolute_percentage_error, name=name, **kwargs)


@keras_export('keras.metrics.log_gaussian_error',
              'keras.losses.log_gaussian_error')
@dispatch.add_dispatch_support
def log_gaussian_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    reshaped = [-1] + y_true.shape.as_list()[1:]
    mu = tf.reshape(y_pred[..., 0], shape=reshaped)
    sigma = tf.reshape(y_pred[..., 1], shape=reshaped)
    loss = 0.5 * math_ops.log(math_ops.sqrt(2 * math.pi)) \
         + 0.5 * math_ops.log(math_ops.square(sigma)) \
         + math_ops.truediv(math_ops.square(y_true - mu), 2 * math_ops.square(sigma))
    return math_ops.reduce_mean(math_ops.square(loss))


@keras_export('keras.metrics.symmetric_mean_absolute_percentage_error',
              'keras.metrics.smape',
              'keras.metrics.SMAPE',
              'keras.losses.symmetric_mean_absolute_percentage_error',
              'keras.losses.smape',
              'keras.losses.SMAPE')
@dispatch.add_dispatch_support
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.abs(y_pred - y_true) / \
           K.maximum((math_ops.abs(y_true) + math_ops.abs(y_pred)), K.epsilon())
    return 2.0 * K.mean(diff, axis=-1)


losses_custom_objects = {
    'LogGaussianLoss': LogGaussianLoss,
    'log_gaussian_error': log_gaussian_error,
    'SymmetricMeanAbsolutePercentageError': SymmetricMeanAbsolutePercentageError,
    'symmetric_mean_absolute_percentage_error': symmetric_mean_absolute_percentage_error,
}