import tensorflow as tf

tf_version = str(tf.__version__)

if int(tf_version.split(".")[1]) < 10:
    from tensorflow.keras.optimizers import *
else:
    from tensorflow.keras.optimizers.legacy import *

from ._optimizers import AdamP

from ._optimizers import optimizer_custom_objects