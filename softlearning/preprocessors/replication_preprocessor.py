import tensorflow as tf
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableSequential
from softlearning.utils.tensorflow import nest

tfkl = tf.keras.layers

def replication_preprocessor(
        n=1,
        scale_factor=1,
        name='replication_preprocessor'):

    if n == 0:
        return tfkl.Flatten()

    def replicate_and_scale(x):
        x = tf.tile(x, [1, n])
        x = scale_factor * x
        return x

    return tfkl.Lambda(replicate_and_scale)
