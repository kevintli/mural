import numpy as np
import tensorflow as tf
from gym import spaces

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

from softlearning.utils.keras import PicklableKerasModel
from .base_preprocessor import BasePreprocessor
from .normalization import (
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization)


def convnet(input_shape,
            output_size,
            conv_filters=(64, 64, 64),
            conv_kernel_sizes=(3, 3, 3),
            conv_strides=(2, 2, 2),
            use_global_average_pool=False,
            normalization_type=None,
            normalization_kwargs={},
            downsampling_type='conv',
            name='convnet',
            *args,
            **kwargs):
    assert downsampling_type in ('pool', 'conv'), downsampling_type

    img_input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = img_input

    for (conv_filter, conv_kernel_size, conv_stride) in zip(
            conv_filters, conv_kernel_sizes, conv_strides):
        x = layers.Conv2D(
            filters=conv_filter,
            kernel_size=conv_kernel_size,
            strides=(conv_stride if downsampling_type == 'conv' else 1),
            padding="SAME",
            activation='linear',
            *args,
            **kwargs
        )(x)

        if normalization_type == 'batch':
            x = layers.BatchNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'layer':
            x = LayerNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'group':
            x = GroupNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'instance':
            x = InstanceNormalization(**normalization_kwargs)(x)
        elif normalization_type == 'weight':
            raise NotImplementedError(normalization_type)
        else:
            assert normalization_type is None, normalization_type

        x = layers.LeakyReLU()(x)

        if downsampling_type == 'pool' and conv_stride > 1:
            x = getattr(tf.keras.layers, 'AvgPool2D')(
                pool_size=conv_stride, strides=conv_stride
            )(x)

    if use_global_average_pool:
        x = layers.GlobalAveragePooling2D(name='average_pool')(x)
    else:
        x = tf.keras.layers.Flatten()(x)

    model = models.Model(img_input, x, name=name)
    model.summary()
    return model


def convnet_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        name="convnet_preprocessor",
        make_picklable=True,
        *args,
        **kwargs):
    inputs = [
        layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    image_size = np.prod(image_shape)
    images_flat, input_raw = layers.Lambda(
        lambda x: [x[..., :image_size], x[..., image_size:]]
    )(concatenated_input)

    images = layers.Reshape(image_shape)(images_flat)
    preprocessed_images = convnet(
        input_shape=image_shape,
        output_size=output_size - input_raw.shape[-1],
        *args,
        **kwargs,
    )(images)
    output = layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([preprocessed_images, input_raw])

    preprocessor = PicklableKerasModel(inputs, output, name=name)

    return preprocessor


class ConvnetPreprocessor(BasePreprocessor):
    def __init__(self, observation_space, output_size, *args, **kwargs):
        super(ConvnetPreprocessor, self).__init__(
            observation_space, output_size)

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self._convnet = convnet_preprocessor(
            input_shapes=input_shapes,
            output_size=output_size,
            *args,
            **kwargs)

    def transform(self, observation):
        transformed = self._convnet(observation)
        return transformed
