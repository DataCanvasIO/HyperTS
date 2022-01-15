import math

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.losses.LogGaussianLoss')
class LogGaussianLoss(losses.Loss):
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
        super(LogGaussianLoss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        mu, sigma = tf.split(y_pred, 2, axis=-1)
        return tf.reduce_mean(
            math_ops.log(tf.math.sqrt(2 * math.pi))
            + math_ops.log(sigma)
            + math_ops.truediv(tf.math.square(y_true - mu), 2 * math_ops.square(sigma)))

    def get_config(self):
        return super(LogGaussianLoss, self).get_config()


@keras_export('keras.metrics.symmetric_mean_absolute_percentage_error',
              'keras.metrics.smape',
              'keras.metrics.SMAPE',
              'keras.losses.symmetric_mean_absolute_percentage_error',
              'keras.losses.smape',
              'keras.losses.SMAPE')
@dispatch.add_dispatch_support
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.abs(y_true - y_pred) / \
           K.maximum((math_ops.abs(y_true) + math_ops.abs(y_pred)), K.epsilon())
    return 2.0 * 100. * K.mean(diff, axis=-1)


losses_custom_objects = {
    'LogGaussianLoss': LogGaussianLoss,
    'symmetric_mean_absolute_percentage_error': symmetric_mean_absolute_percentage_error,
}