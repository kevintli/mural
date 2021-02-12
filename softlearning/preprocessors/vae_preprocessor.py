from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.keras import regularizers
from softlearning.utils.keras import PicklableModel

tfk = tf.keras
tfkl = tf.keras.layers


def create_vae_encoder_model(image_shape,
                             latent_dim,
                             trainable=True,
                             kernel_regularizer=regularizers.l2(5e-4),
                             extra_input_shape=None,
                             name='encoder'):
    conv2d = functools.partial(
        tfkl.Conv2D,
        kernel_size=3,
        activation=tfkl.LeakyReLU(),
        trainable=trainable,
        kernel_regularizer=kernel_regularizer,
        padding='SAME',
    )

    def preprocess(x):
        return tf.image.convert_image_dtype(x, tf.float32)

    # Functional model, need to debug this
    x = tfkl.Input(shape=image_shape, name='pixel_input')
    preprocessed_x = tfkl.Lambda(preprocess)(x)
    # TODO: Be able to specify convnet params, do this with `convnet.py`
    conv_output_0 = conv2d(filters=64, strides=2)(preprocessed_x)
    conv_output_1 = conv2d(filters=64, strides=2)(conv_output_0)
    # conv_output_2 = conv2d(filters=32, strides=1)(conv_output_1)
    conv_output_2 = conv2d(filters=32, strides=2)(conv_output_1)
    output = tfkl.Flatten()(conv_output_2)

    if extra_input_shape:
        s = tfkl.Input(shape=extra_input_shape, name='extra_input')
        output = tfkl.concatenate([output, s])

    output = tfkl.Dense(
        2 * latent_dim,
        trainable=trainable,
        kernel_regularizer=kernel_regularizer
    )(output)

    mean, logvar = tfkl.Lambda(
        lambda mean_logvar_concat: tf.split(
            mean_logvar_concat,
            num_or_size_splits=2,
            axis=-1
        )
    )(output)

    def sample(inputs):
        """Reparameterization that is batch_size and dimension agnostic."""
        z_mean, z_logvar = inputs
        batch_size = tf.shape(z_mean)[0]
        dim = tf.keras.backend.int_shape(z_mean)[1]
        eps = tf.random.normal(shape=(batch_size, dim))
        return eps * tf.exp(z_logvar * 0.5) + z_mean # , (batch_size, dim, eps)

    latents = tfkl.Lambda(
        sample, output_shape=(latent_dim, ), name='z'
    )([mean, logvar])

    if extra_input_shape:
        return tfk.Model([x, s], output, name=name)
    else:
        return tfk.Model(x, [mean, logvar, latents], name=name)

def create_vae_decoder_model(latent_dim,
                             trainable=True,
                             kernel_regularizer=regularizers.l2(5e-4),
                             name='decoder'):
    conv2d_transpose = functools.partial(
        tfkl.Conv2DTranspose,
        kernel_size=3,
        activation=tfkl.LeakyReLU(),
        trainable=trainable,
        kernel_regularizer=kernel_regularizer,
        padding='SAME'
    )
    return tfk.Sequential([
        tfkl.InputLayer(input_shape=(latent_dim,)),
        # This layer expands the dimensionality a lot.
        tfkl.Dense(
            units=4*4*32,
            activation=tfkl.LeakyReLU(),
            trainable=trainable,
            kernel_regularizer=kernel_regularizer
        ),
        tfkl.Reshape(target_shape=(4, 4, 32)),
        conv2d_transpose(filters=64, strides=2),
        conv2d_transpose(filters=64, strides=2),
        conv2d_transpose(filters=32, strides=2),
        conv2d_transpose(filters=3, strides=1),
    ], name=name)


def create_vae(image_shape,
               latent_dim,
               *args,
               beta=1.0,
               name='beta_vae',
               **kwargs):
    encoder = create_vae_encoder_model(
        image_shape=image_shape,
        latent_dim=latent_dim,
        **kwargs)
    decoder = create_vae_decoder_model(latent_dim, **kwargs)

    outputs = decoder(encoder(encoder.inputs)[2])
    vae = PicklableModel(encoder.inputs, outputs, name=name)
    vae.beta = beta
    vae.encoder = encoder
    vae.decoder = decoder
    vae.latent_dim = latent_dim

    return vae


class OnlineVAEPreprocessor(tfk.Model):
    def __init__(self,
                 image_shape,
                 latent_dim=32,
                 beta=1.0,
                 encoder_config=None,
                 decoder_config=None,
                 kernel_regularizer=regularizers.l2(5e-4),
                 name='online_vae_preprocessor',
                 *args,
                 **kwargs):
        super(OnlineVAEPreprocessor, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.beta = beta
        if encoder_config:
            self.encoder = PicklableModel.from_config(encoder_config)
        else:
            self.encoder = create_vae_encoder_model(
                image_shape=image_shape,
                latent_dim=latent_dim,
                **kwargs)
        if decoder_config:
            self.decoder = PicklableModel.from_config(decoder_config)
        else:
            self.decoder = create_vae_decoder_model(latent_dim, **kwargs)

    def call(self, inputs, include_reconstructions=False):
        z_mean, z_logvar, z = self.encoder(inputs)
        outputs = [z_mean, z_logvar, z]
        if include_reconstructions:
            reconstructions = self.decoder(z)
            outputs += [reconstructions]
        return outputs

    def get_config(self):
        config = {
            'cls': self.__class__,
            'image_shape': self.image_shape,
            'latent_dim': self.latent_dim,
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config()
        }
        return config

    @classmethod
    def from_config(self, config):
        vae_preprocessor = config['cls'](
            image_shape=config['image_shape'],
            latent_dim=config['latent_dim'],
            encoder_config=config['encoder'],
            decoder_config=config['decoder'],
        )
        return vae_preprocessor

# class OnlineVAEPreprocessor:
#     def __init__(self,
#                  image_shape,
#                  latent_dim=32,
#                  kernel_regularizer=regularizers.l2(5e-4),
#                  name='online_vae_preprocessor',
#                  *args,
#                  **kwargs):
#         self.image_shape = image_shape

#         self.vae = create_vae(
#             image_shape,
#             latent_dim,
#             **kwargs)
#         # Take the mean, not the sampled latent, since gradients won't pass.
#         # Should I be doing a stop gradient here?
#         self.preprocessor = PicklableModel(
#             self.vae.input,
#             tf.stop_gradient(self.vae.encoder.outputs[0]),
#             name=name)

#         self.name = name

#     def __call__(self, x):
#         return self.preprocessor(x)

#     def get_config(self):
#         config = {
#             'cls': self.__class__,
#             'image_shape': self.image_shape,
#             'latent_dim': self.vae.latent_dim,
#             'beta': self.vae.beta,
#             'vae': self.vae.get_config(),
#             'preprocessor': self.preprocessor.get_config(),
#         }
#         return config

#     @classmethod
#     def from_config(cls, config):
#         vae_preprocessor = OnlineVAEPreprocessor(config['image_shape'])
#         vae_preprocessor.preprocessor = PicklableModel.from_config(config['preprocessor'])
#         vae = vae_preprocessor.vae = PicklableSequential.from_config(config['preprocessor'])
#         vae.encoder = vae.get_layer('encoder')
#         vae.decoder = vae.get_layer('decoder')
#         vae.latent_dim = config['latent_dim']
#         vae.beta = config['beta']
#         return vae_preprocessor

#     @property
#     def trainable_variables(self):
#         return self.vae.trainable_variables
