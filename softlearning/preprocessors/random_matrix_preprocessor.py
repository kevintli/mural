import tensorflow as tf
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableSequential
from softlearning.utils.tensorflow import nest

tfkl = tf.keras.layers

def random_matrix_preprocessor(
        output_size_scale_factor=1,
        coefficient_range=(-1., 1.),
        name='random_matrix_preprocessor'):
    """
    ouptut_size_scale_factor: int
        The factor by which the matrix multiply will scale the input observation.
        For example, if it equals 2, then the matrix will be n x 2n, where n is
        the dimension of the intput observation.
    coefficient_range: np.ndarray (2,)
        The range from which coefficients will be sampled from uniformly.
        This will determine the coefficients of the linear combinations.
    """
    output_size_scale_factor = int(output_size_scale_factor)
    # Save the random matrix in this scope, so that the inputs to the
    # policy are not stochastic. Want the same fixed matrix mupliy every
    # time.
    random_matrix = None
    def random_matrix_multiply(x):
        nonlocal random_matrix
        n = x.shape[1].value
        if random_matrix is None:
            random_matrix = tf.random.uniform(
                shape=(n, output_size_scale_factor * n),
                minval=coefficient_range[0],
                maxval=coefficient_range[1],
                dtype=tf.float32)
        x = tf.matmul(x, random_matrix)
        return x

    return tfkl.Lambda(random_matrix_multiply)
