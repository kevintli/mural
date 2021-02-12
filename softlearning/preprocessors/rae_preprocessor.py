
import tensorflow as tf
from tensorflow.keras import regularizers

from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model, convnet_transpose_model

tfk = tf.keras
tfkl = tf.keras.layers


def create_encoder_model(input_shape,
                         latent_dim,
                         trainable=True,
                         kernel_regularizer_type='l2',
                         kernel_regularizer_lambda=5e-4,
                         name='encoder',
                         **kwargs):
    x = tfkl.Input(shape=input_shape, name='pixel_input')
    kernel_regularizer = REGULARIZERS[kernel_regularizer_type](
        kernel_regularizer_lambda)
    encoder = convnet_model(
        # preprocessed_image_range=(0, 1), # Preprocess images to [0, 1]
        kernel_regularizer=kernel_regularizer,
        **kwargs)
    fc = tfkl.Dense(latent_dim, activation='linear')
    z = fc(encoder(x))
    return PicklableModel(x, z, name=name)


REGULARIZERS = {
    'l1': regularizers.l1,
    'l2': regularizers.l2,
    'l1_l2': regularizers.l1_l2,
}


def create_decoder_model(latent_dim,
                         output_shape,
                         trainable=True,
                         kernel_regularizer_type='l2',
                         kernel_regularizer_lambda=5e-4,
                         name='decoder',
                         **kwargs):
    assert kernel_regularizer_type in REGULARIZERS, (
        f'Regularizer type must be one of {str(REGULARIZERS.keys())}')
    z = tfkl.Input(shape=(latent_dim, ), name='latent_input')
    kernel_regularizer = REGULARIZERS[kernel_regularizer_type](
        kernel_regularizer_lambda)
    decoder = convnet_transpose_model(
        output_shape=output_shape,
        kernel_regularizer_type=kernel_regularizer,
        output_activation='sigmoid', # Want [0, 1] outputs
        **kwargs
    )
    return PicklableModel(z, decoder(z), name=name)


def create_rae(image_shape,
               latent_dim,
               name='regularized_ae',
               **kwargs):
    encoder = create_encoder_model(image_shape, latent_dim, **kwargs)
    decoder = create_decoder_model(latent_dim, image_shape, **kwargs)

    z = encoder(encoder.inputs)
    reconstruction = decoder(z)
    rae = PicklableModel(
        encoder.inputs,
        [z, reconstruction],
        name=name
    )

    # rae = PicklableModel(
    #     encoder.inputs,
    #     decoder(encoder(encoder.inputs)),
    #     name=name)

    # rae.encoder = encoder
    # rae.decoder = decoder
    # rae.latent_dim = latent_dim

    return rae

class RAEPreprocessor(PicklableModel):
    def __init__(self,
                 image_shape,
                 latent_dim,
                 name='rae_preprocessor',
                 **kwargs):
        super(RAEPreprocessor, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.encoder = create_encoder_model(image_shape, latent_dim, **kwargs)
        self.decoder = create_decoder_model(latent_dim, image_shape, **kwargs)

    def call(self, inputs, include_reconstructions=False):
        z = self.encoder(inputs)
        if include_reconstructions:
            reconstructions = self.decoder(z)
            return [z, reconstructions]
        return z

    def get_config(self):
        config = {
            'cls': self.__class__,
            'image_shape': self.image_shape,
            'latent_dim': self.latent_dim,
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'weights': self.get_weights(),
        }
        return config

    @classmethod
    def from_config(self, config):
        assert 'encoder' in config and 'decoder' in config, (
            f'Need to speciy encoder and decoder configs, {config}')
        rae_preprocessor = config['cls'](
            image_shape=config['image_shape'],
            latent_dim=config['latent_dim'])
        rae_preprocessor.encoder = PicklableModel.from_config(config['encoder'])
        rae_preprocessor.decoder = PicklableModel.from_config(config['decoder'])
        rae_preprocessor.set_weights(config['weights'])
        return rae_preprocessor

# TODO: Create an abstract interface for preprocessors
# class RAEPreprocessor:
#     def __init__(self,
#                  image_shape,
#                  latent_dim=32,
#                  name='rae_preprocessor',
#                  **kwargs):
#         self.image_shape = image_shape
#         self.rae = create_rae(image_shape, latent_dim, **kwargs)
#         self.preprocessor = PicklableModel(
#             self.rae.inputs,
#             tf.stop_gradient(self.rae.encoder.output),
#             name=name
#         )
#         self.name = name

#     def __call__(self, x):
#         return self.preprocessor(x)

#     @property
#     def trainable_variables(self):
#         return self.rae.trainable_variables
