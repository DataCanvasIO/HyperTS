"""Utils related to keras model saving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import generic_utils


def compile_args_from_training_config(training_config, custom_objects=None):
  """Return model.compile arguments from training config."""
  if custom_objects is None:
    custom_objects = {}

  with generic_utils.CustomObjectScope(custom_objects):
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config)

    # Recover losses.
    loss = None
    loss_config = training_config.get('loss', None)
    if loss_config is not None:
      loss = _deserialize_nested_config(losses.deserialize, loss_config)

    # Recover metrics.
    metrics = None
    metrics_config = training_config.get('metrics', None)
    if metrics_config is not None:
      metrics = _deserialize_nested_config(_deserialize_metric, metrics_config)

    # Recover weighted metrics.
    weighted_metrics = None
    weighted_metrics_config = training_config.get('weighted_metrics', None)
    if weighted_metrics_config is not None:
      weighted_metrics = _deserialize_nested_config(_deserialize_metric,
                                                    weighted_metrics_config)

    sample_weight_mode = training_config['sample_weight_mode'] if hasattr(
        training_config, 'sample_weight_mode') else None
    loss_weights = training_config['loss_weights']

  return dict(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)


def _deserialize_nested_config(deserialize_fn, config):
  """Deserializes arbitrary Keras `config` using `deserialize_fn`."""

  def _is_single_object(obj):
    if isinstance(obj, dict) and 'class_name' in obj:
      return True  # Serialized Keras object.
    if isinstance(obj, six.string_types):
      return True  # Serialized function or string.
    return False

  if config is None:
    return None
  if _is_single_object(config):
    return deserialize_fn(config)
  elif isinstance(config, dict):
    return {
        k: _deserialize_nested_config(deserialize_fn, v)
        for k, v in config.items()
    }
  elif isinstance(config, (tuple, list)):
    return [_deserialize_nested_config(deserialize_fn, obj) for obj in config]

  raise ValueError('Saved configuration not understood.')


def _deserialize_metric(metric_config):
  """Deserialize metrics, leaving special strings untouched."""
  from tensorflow.python.keras import metrics as metrics_module  # pylint:disable=g-import-not-at-top
  if metric_config in ['accuracy', 'acc', 'crossentropy', 'ce']:
    # Do not deserialize accuracy and cross-entropy strings as we have special
    # case handling for these in compile, based on model output shape.
    return metric_config
  return metrics_module.deserialize(metric_config)