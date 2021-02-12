import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers
import numpy as np

from softlearning.utils.keras import PicklableSequential
from softlearning.models.normalization import (
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization)
from softlearning.utils.tensorflow import nest


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


"""
Convnet customization options
"""

POOLING_TYPES = {
    'avg_pool': layers.AvgPool2D,
    'max_pool': layers.MaxPool2D,
}

OUTPUT_TYPES = (
    'spatial_softmax',
    'dense',
    'flatten'
)
DEFAULT_OUTPUT_KWARGS = {'type': 'flatten'}

NORMALIZATION_TYPES = {
    'batch': layers.BatchNormalization,
    'layer': LayerNormalization,
    'group': GroupNormalization,
    'instance': InstanceNormalization,
    None: None,
}

REGULARIZERS = {
    'l1': regularizers.l1,
    'l2': regularizers.l2,
    'l1_l2': regularizers.l1_l2,
    None: None,
}


def convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        padding="SAME",
        preprocessed_image_range=(-1, 1),
        normalization_type=None,
        normalization_kwargs={},
        downsampling_type='conv',
        activation=layers.LeakyReLU,
        output_kwargs=None,
        kernel_regularizer=None,
        name="convnet",
        *args,
        **kwargs):
    normalization_layer = NORMALIZATION_TYPES[normalization_type]

    def conv_block(conv_filter,
                   conv_kernel_size,
                   conv_stride,
                   name='conv_block'):
        block_parts = [
            layers.Conv2D(
                *args,
                filters=conv_filter,
                kernel_size=conv_kernel_size,
                strides=(conv_stride if downsampling_type == 'conv' else 1),
                padding=padding,
                activation='linear',
                kernel_regularizer=kernel_regularizer,
                **kwargs),
        ]

        if normalization_layer is not None:
            block_parts += [normalization_layer(**normalization_kwargs)]

        block_parts += [(layers.Activation(activation)
                         if isinstance(activation, str)
                         else activation())]

        if downsampling_type in POOLING_TYPES:
            block_parts += [
                POOLING_TYPES[downsampling_type](
                    pool_size=conv_stride, strides=conv_stride
                )
            ]

        block = tfk.Sequential(block_parts, name=name)
        return block

    assert (
        len(preprocessed_image_range) == 2 and
        preprocessed_image_range[0] < preprocessed_image_range[1]), (
        'Preprocessed image range must be of the form (a, b), where a < b'
    )
    low, high = preprocessed_image_range

    def preprocess(x):
        """Cast to float, normalize, and concatenate images along last axis."""
        import tensorflow as tf
        x = nest.map_structure(
            lambda image: tf.image.convert_image_dtype(image, tf.float32), x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        # x = (tf.image.convert_image_dtype(x, tf.float32) - 0.5) * 2.0
        # TODO: Why is the image being converted to float32 twice? Once in the
        # nest and once down here?
        x = (high - low) * tf.image.convert_image_dtype(x, tf.float32) + low
        return x

    output_kwargs = output_kwargs or DEFAULT_OUTPUT_KWARGS
    output_type = output_kwargs.get('type', DEFAULT_OUTPUT_KWARGS['type'])
    if output_type == 'spatial_softmax':
        def spatial_softmax(x):
            # Create learnable temperature parameter `alpha`
            alpha = tf.Variable(1., dtype=tf.float32, name='softmax_alpha')
            width, height, channels = x.shape[1:]
            x_flattened = tf.reshape(
                x, [-1, width * height, channels])
            softmax_attention = tf.math.softmax(x_flattened / alpha, axis=1)
            # TODO: Fix this; redundant, since I'm going to reflatten it later
            softmax_attention = tf.reshape(
                softmax_attention, [-1, width, height, channels])
            return softmax_attention

        def calculate_expectation(distributions):
            width, height, channels = distributions.shape[1:]

            # Create matrices where all xs/ys are the same value acros
            # the row/col. These will be multiplied by the softmax distr
            # to get the 2D expectation.
            pos_x, pos_y = tf.meshgrid(
                tf.linspace(-1., 1., num=width),
                tf.linspace(-1., 1., num=height),
                indexing='ij'
            )
            # Reshape to a column vector to satisfy multiply broadcast.
            pos_x, pos_y = (
                tf.reshape(pos_x, [-1, 1]),
                tf.reshape(pos_y, [-1, 1])
            )

            distributions = tf.reshape(
                distributions, [-1, width * height, channels])

            expected_x = tf.math.reduce_sum(
                pos_x * distributions, axis=[1], keepdims=True)
            expected_y = tf.math.reduce_sum(
                pos_y * distributions, axis=[1], keepdims=True)
            expected_xy = tf.concat([expected_x, expected_y], axis=1)
            feature_keypoints = tf.reshape(expected_xy, [-1, 2 * channels])
            return feature_keypoints

        output_layer = tfk.Sequential([
            tfkl.Lambda(spatial_softmax),
            tfkl.Lambda(calculate_expectation)
        ])
    elif output_type == 'dense':
        # TODO: Implement this with `feedforward` network
        pass
    else:
        output_layer = tfkl.Flatten()

    model = PicklableSequential((
        tfkl.Lambda(preprocess, name='preprocess'),
        *[
            conv_block(
                conv_filter,
                conv_kernel_size,
                conv_stride,
                name=f'conv_block_{i}')
            for i, (conv_filter, conv_kernel_size, conv_stride) in
            enumerate(zip(conv_filters, conv_kernel_sizes, conv_strides))
        ],
        output_layer,
    ), name=name)
    return model


def convnet_transpose_model(
        output_shape=(32, 32, 3),
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        padding="SAME",
        normalization_type=None,
        normalization_kwargs={},
        downsampling_type='conv',
        activation=layers.LeakyReLU,
        output_activation='tanh',
        kernel_regularizer=None,
        name='convnet_transpose',
        *args,
        **kwargs):

    normalization_layer = NORMALIZATION_TYPES[normalization_type]
    # kernel_regularizer = REGULARIZERS[kernel_regularizer_type]

    def conv_transpose_block(n_filters,
                             kernel_size,
                             stride,
                             block_activation,
                             name='conv_transpose_block'):
        conv_stride = stride if downsampling_type == 'conv' else 1
        block_parts = [
            tfkl.Conv2DTranspose(
                filters=n_filters,
                kernel_size=kernel_size,
                padding=padding,
                strides=conv_stride,
                activation='linear',
                kernel_regularizer=kernel_regularizer,
            )
        ]
        if normalization_layer is not None:
            block_parts += [normalization_layer(**normalization_kwargs)]

        block_parts += [(layers.Activation(block_activation, name=block_activation)
                         if isinstance(block_activation, str)
                         else block_activation())]

        if downsampling_type in POOLING_TYPES:
            block_parts += [
                POOLING_TYPES[downsampling_type](
                    pool_size=stride, strides=stride
                )
            ]
        block = tfk.Sequential(block_parts, name=name)
        return block

    assert len(output_shape) == 3, 'Output shape needs to be (w, h, c), w = h'
    w, h, c = output_shape

    # TODO: generalize this to diffenent padding types (only works for
    # SAME right now) as well as different sized images (this only really
    # works for nice powers of stride length), i.e 32 -> 16 -> 8 -> 4
    if padding != 'SAME':
        raise NotImplementedError

    base_w = w // np.product(conv_strides)
    base_h = h // np.product(conv_strides)
    base_shape = (base_w, base_h, conv_filters[0])

    model = PicklableSequential([
        tfkl.Dense(
            units=np.product(base_shape),
            # activation=(layers.Activation(activation)
            #             if isinstance(activation, str)
            #             else activation()),
            kernel_regularizer=kernel_regularizer
        ),
        (layers.Activation(activation)
         if isinstance(activation, str)
         else activation()),
        tfkl.Reshape(target_shape=base_shape),
        *[
            conv_transpose_block(
                conv_filter,
                conv_kernel_size,
                conv_stride,
                activation,
                name=f'conv_transpose_block_{i}')
            for i, (conv_filter, conv_kernel_size, conv_stride) in
            enumerate(zip(conv_filters, conv_kernel_sizes, conv_strides))
        ],
        conv_transpose_block(
            n_filters=c,
            kernel_size=conv_kernel_sizes[-1],
            stride=1,
            block_activation=output_activation,
            name=f'conv_transpose_block_output',
        ),
    ], name=name)
    return model
