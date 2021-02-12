import tensorflow as tf
import numpy as np
import functools
import gzip
import glob
import pickle
from tensorflow.keras import regularizers
from softlearning.utils.keras import PicklableModel
tfk = tf.keras
tfkl = tf.keras.layers

# tf.enable_eager_execution()

"""
Training methods
"""


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_elbo_loss(model, x, beta=1.0):
    if isinstance(x, tuple):
        image, claw_state = x
        mean, logvar = model.encode(image)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(tf.concat([z, claw_state], axis=-1))
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=model.preprocess(image))

    else:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=model.preprocess(x))

    # Cross entropy reconstruction loss assumes that the pixels
    # are all independent Bernoulli r.v.s
    # Need to preprocess the label, so the output will be normalized.
    # Sum across all pixels (row/col) + channels
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # Calculate the KL divergence (difference between log of unit
    # normal prior and posterior)
    logpz = log_normal_pdf(z, 0., 0.)  # Prior PDF
    logqz_x = log_normal_pdf(z, mean, logvar)  # Posterior
    reconstruction_loss = logpx_z
    kl_divergence = logpz - logqz_x
    loss = reconstruction_loss + beta * kl_divergence
    return -tf.reduce_mean(loss)


def compute_elbo_loss_split(model, x, beta=1.0):
    if isinstance(x, tuple):
        image, claw_state = x
        mean, logvar = model.encode(image)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(tf.concat([z, claw_state], axis=-1))
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=model.preprocess(image))

    else:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=model.preprocess(x))

    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)  # Prior PDF
    logqz_x = log_normal_pdf(z, mean, logvar)  # Posterior
    reconstruction_loss = logpx_z
    kl_divergence = logpz - logqz_x
    loss = reconstruction_loss + beta * kl_divergence
    return (
        -tf.reduce_mean(reconstruction_loss),
        -tf.reduce_mean(beta * kl_divergence)
    )


@tf.function
def compute_apply_gradients(model, x, optimizer, beta=1.0):
    with tf.GradientTape() as tape:
        loss = compute_elbo_loss(model, x, beta=beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


"""
VAE model definition
"""

class VAE(tfk.Model):
    def __init__(
            self,
            image_shape,
            latent_dim=16,
            beta=1.0,
            extra_input_shape=(0,),
            kernel_regularizer=regularizers.l2(l=5e-4),
            optimizer=tf.keras.optimizers.Adam(1e-4)):
        super().__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.encoder = self.create_encoder_model(
            extra_input_shape=extra_input_shape)
        self.decoder = self.create_decoder_model(
            extra_input_shape=extra_input_shape)
        self.beta = beta

        # Set up variables for online training
        self.optimizer = optimizer
        self.elbo_history = []
        self.kl_history = []
        self.reconstruct_loss_history = []

    # def compute_elbo_loss(self, x):
    #     mean, logvar = self.encode(x)
    #     z = self.reparameterize(mean, logvar)
    #     x_logit = self.decode(z)
    #     # Cross entropy reconstruction loss assumes that the pixels
    #     # are all independent Bernoulli r.v.s
    #     # Need to preprocess the label, so the output will be normalized.
    #     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=x_logit, labels=self.preprocess(x))
    #     # Sum across all pixels (row/col) + channels
    #     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    #     # Calculate the KL divergence (difference between log of unit
    #     # normal prior and posterior)
    #     logpz = log_normal_pdf(z, 0., 0.)  # Prior PDF
    #     logqz_x = log_normal_pdf(z, mean, logvar)  # Posterior
    #     reconstruction_loss = logpx_z
    #     kl_divergence = logpz - logqz_x
    #     loss = reconstruction_loss + self.beta * kl_divergence
    #     return -tf.reduce_mean(loss)

    # def train_iter(self, x):
    #     """
    #     Performs one step of gradient descent on the batch `x`.
    #     """
    #     with tf.GradientTape() as tape:
    #         loss = self.compute_elbo_loss(x)
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(
    #         zip(gradients, self.trainable_variables))
    #     return loss

    def preprocess(self, x):
        """
        Turn integers into floats normalized between [0, 1]
        """
        return tf.image.convert_image_dtype(x, tf.float32)

    def create_encoder_model(self,
                             image_shape=None,
                             latent_dim=None,
                             trainable=True,
                             kernel_regularizer=None,
                             use_functional_model=False,
                             extra_input_shape=(0,),
                             # padding='VALID', # CHANGE THIS BACK FOR OLD MODELS
                             padding='SAME',
                             name='encoder'):
        if image_shape is None:
            image_shape = self.image_shape
        if latent_dim is None:
            latent_dim = self.latent_dim
        conv2d = functools.partial(
            tfkl.Conv2D,
            kernel_size=3,
            activation=tfkl.LeakyReLU(),
            trainable=trainable,
            kernel_regularizer=kernel_regularizer,
            padding=padding,
        )
        if use_functional_model:
            # Functional model, need to debug this
            x = tfkl.Input(shape=image_shape, name='pixel_input')
            preprocessed_x = tfkl.Lambda(self.preprocess)(x)
            conv_output = conv2d(filters=64, strides=2)(preprocessed_x)
            conv_output = conv2d(filters=64, strides=2)(conv_output)
            # conv_output = conv2d(filters=32, strides=1)(conv_output) # new layer
            conv_output = conv2d(filters=32, strides=2)(conv_output)
            # conv_output_3 = conv2d(filters=32, strides=2)(conv_output_2)
            # conv_output_2 = conv2d(filters=32, strides=2)(conv_output_1)
            output = tfkl.Flatten()(conv_output)
            output = tfkl.Dense(
                latent_dim + latent_dim,
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            )(output)

            # mean, logvar = tfkl.Lambda(
            #     lambda mean_logvar_concat: tf.split(
            #         mean_logvar_concat,
            #         num_or_size_splits=2,
            #         axis=-1
            #     )
            # )

            # def sampling(inputs):
            #     """Reparameterization that is batch_size and dimension agnostic."""
            #     z_mean, z_logvar = inputs
            #     batch_size = tf.shape(z_mean, 0)
            #     dim = tf.keras.backend.int_shape(z_mean)[1]
            #     eps = tf.random.normal(size=(batch_size, dim))
            #     return eps * tf.exp(z_logvar * 0.5) + z_mean

            # latents = tfkl.Lambda(
            #     sampling, output_shape=(latent_dim, ), name='z'
            # )([mean, logvar])
            if extra_input_shape != (0,):
                s = tfkl.Input(shape=extra_input_shape, name='extra_input')
                output = tfkl.concatenate([output, s])
                return tfk.Model([x, s], output, name=name)
            else:
                return tfk.Model(x, [mean, logvar, latents], name=name)
        else:
            # return tfk.Model(x, output, name=name)

            layers = [
                tfkl.InputLayer(input_shape=image_shape, name='pixel_input'),
                tfkl.Lambda(self.preprocess),
                conv2d(filters=64, strides=2),
                conv2d(filters=64, strides=2),
                conv2d(filters=32, strides=2),
                # conv2d(filters=32, strides=2),
                tfkl.Flatten(),
                # === JUST ADDED A NEW DENSE LAYER ===
                tfkl.Dense(
                    4 * latent_dim, # 128 for 32-dim latent
                    trainable=trainable,
                    kernel_regularizer=kernel_regularizer,
                ),
                tfkl.Dense(
                    latent_dim + latent_dim,
                    trainable=trainable,
                    kernel_regularizer=kernel_regularizer
                )
            ]
            # TODO: Use PicklableSequential
            model =  tfk.Sequential(layers, name=name)
            # if extra_input_shape:
            #     s = tfkl.Input(shape=extra_input_shape, name='extra_input')
            #     return tfkl.concatenate([model, s])
            # else:
            return model

    def create_decoder_model(self,
                             latent_dim=None,
                             trainable=True,
                             kernel_regularizer=None,
                             extra_input_shape=(0,),
                             name='decoder'):
        if latent_dim is None:
            latent_dim = self.latent_dim

        conv2d_transpose = functools.partial(
            tfkl.Conv2DTranspose,
            kernel_size=3,
            activation=tfkl.LeakyReLU(),
            trainable=trainable,
            kernel_regularizer=kernel_regularizer,
            padding='SAME'
        )
        return tfk.Sequential([
            tfkl.InputLayer(input_shape=(latent_dim + extra_input_shape[0],),
                            name='latent_input'),
            tfkl.Dense(
                units=2*2*32,
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
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
            # conv2d_transpose(filters=32, strides=2),
            conv2d_transpose(filters=3, strides=1),
        ], name=name)

        # return tfk.Sequential([
        #     tfkl.InputLayer(input_shape=(latent_dim,)),
        #     # This layer expands the dimensionality a lot.
        #     tfkl.Dense(
        #         units=4*4*32,
        #         activation=tfkl.LeakyReLU(),
        #         trainable=trainable,
        #         kernel_regularizer=kernel_regularizer
        #     ),
        #     tfkl.Reshape(target_shape=(4, 4, 32)),
        #     conv2d_transpose(filters=64, strides=2),
        #     conv2d_transpose(filters=64, strides=2),
        #     conv2d_transpose(filters=32, strides=2),
        #     conv2d_transpose(filters=3, strides=1),
        # ], name=name)

    # def get_vae_model(self):
    #     encoder = self.encoder
    #     decoder = self.decoder
    #     decoder_outputs = decoder(encoder(encoder.inputs)[2])
    #     vae = PicklableModel(encoder.inputs, decoder_outputs)
    #     vae.beta = self.beta

    def sample(self, eps=None, n_samples=16):
        """
        Return `n_samples` reconstructions from randomly sampled
        latents from the (Gaussian) prior.
        """
        if eps is None:
            eps = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    # def reconstruct(self, x):
    #     """
    #     Forward pass through the VAE.
    #     TODO: Figure out whether this should be the __call__ method instead.
    #     """
    #     # mean, logvar = self.encode(x)
    #     # z = self.reparameterize(mean, logvar)
    #     # return self.decode(z, apply_sigmoid=True)
    #     mean, logvar, z = self.encoder(x) # Is this right?
    #     return self.decode(z, apply_sigmoid=True)

    def encode(self, x, mean_only=False):
        """
        Forward pass through the encoder network.
        Inputs: image `x`, Outputs: latent mean
        """
        mean, logvar = tf.split(
            self.encoder(x), num_or_size_splits=2, axis=1)
        if mean_only:
            return mean
        return mean, logvar

    def __call__(self, x):
        mean = self.encode(x, mean_only=True)
        return mean

    def decode(self, z, apply_sigmoid=False):
        """
        Inputs: latent, Outputs: reconstructions as either
        logits or probabilities. Specify this with `apply_sigmoid`.
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def __call__(self, x):
        mean = self.encode(x, mean_only=True)
        return mean

    def reconstruct(self, x):
        if isinstance(x, tuple):
            image, claw_state = x
            mean, logvar = self.encode(image)
            z = self.reparameterize(mean, logvar)
            x_reconstruct = self.decode(tf.concat([z, claw_state], axis=-1),
                                        apply_sigmoid=True)
        else:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_reconstruct = self.decode(z, apply_sigmoid=True)
        return x_reconstruct

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def reparameterize_split(self, mean_logvar_concat):
        mean, logvar = tf.split(
            mean_logvar_concat, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def get_encoder(self, trainable=True, name='encoder'):
        encoder = self.create_encoder_model(
            self.image_shape, trainable=trainable, name=name)
        # Copy weights over to this new model
        encoder.set_weights(self.encoder.get_weights())
        # Only return the mean (no need to sample an epsilon)

        def get_encoded_mean(mean_logvar_concat):
            mean, logvar = tf.split(
                mean_logvar_concat, num_or_size_splits=2, axis=1)
            return mean
        encoder.add(tfkl.Lambda(get_encoded_mean, name='encoded_mean'))
        encoder.summary()
        return encoder


"""
Training script
"""

def split_data(data, validation_split_ratio=0.1, shuffle=False):
    if shuffle:
        np.random.shuffle(images)

    split_index = int(validation_split_ratio * len(data))
    train_images = images[split_index:]
    test_images = images[:split_index]
    return train_images, test_images


def get_datasets(images,
                 claw_states=None,
                 batch_size=128,
                 # shuffle=True,
                 validation_split_ratio=0.1):
    split_index = int(validation_split_ratio * len(images))
    train_images = images[split_index:]
    test_images = images[:split_index]

    if claw_states is not None:
        train_claw_states = claw_states[split_index:]
        test_claw_states = claw_states[:split_index]
        def train_generator():
            for image, claw_state in zip(train_images, train_claw_states):
                yield image, claw_state

        def test_generator():
            for image, claw_state in zip(test_images, test_claw_states):
                yield image, claw_state
        train_dataset = tf.data.Dataset.from_generator(
                train_generator, output_types=(tf.uint8, tf.float32)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_generator(
                test_generator, output_types=(tf.uint8, tf.float32)).batch(batch_size)

    else:
        def train_generator():
            for image in train_images:
                yield image

        def test_generator():
            for image in test_images:
                yield image
        train_dataset = tf.data.Dataset.from_generator(
                train_generator, tf.uint8).batch(batch_size)
        test_dataset = tf.data.Dataset.from_generator(
                test_generator, tf.uint8).batch(batch_size)

    return train_dataset, test_dataset, train_images, test_images

# if __name__ == '__main__':
#     import argparse
#     import time
#     import os
#     import skimage
#     tf.enable_eager_execution()
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     tf.keras.backend.set_session(sess)

#     REGULARIZER_OPTIONS = {
#         'l1': regularizers.l1,
#         'l2': regularizers.l2,
#         'l1_l2': regularizers.l1_l2,
#     }

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--latent-dim',
#                         type=int,
#                         help='Latent dimension of the VAE',
#                         default=32)
#     parser.add_argument('--image-shape',
#                         type=lambda x: eval(x),
#                         help='(width, height, channels) of the image',
#                         default=(32, 32, 3))
#     parser.add_argument('--save-path-name',
#                         type=str,
#                         help='Name to store the VAE weights under',
#                         default='vae')
#     parser.add_argument('--n-epochs',
#                         type=int,
#                         help='Number of epochs to train for',
#                         default=250)
#     parser.add_argument('--beta',
#                         type=float,
#                         help='Beta parameter for the VAE',
#                         default=1.0)
#     parser.add_argument('--n-examples-to-generate',
#                         type=float,
#                         help='Number of random reconstructions to generate',
#                         default=4)
#     parser.add_argument('--save-weights-frequency',
#                         type=int,
#                         help='Number of epochs between each save',
#                         default=5)
#     parser.add_argument('--train-data-directory',
#                         type=str,
#                         help='.pkl file to load the training data from, formatted as a dictionary with a `pixels` key')
#     parser.add_argument('--eval-data-directory',
#                         type=str,
#                         help='.pkl file to load the evaluation data from, formatted as a dictionary with a `pixels` key')

#     parser.add_argument('--init-weights-directory',
#                         type=str,
#                         help='Directory to load weights from',
#                         default=None)

#     parser.add_argument('--kernel-regularizer',
#                         type=str,
#                         help='Kernel regularizer to use for Conv2D and Dense layers',
#                         choices=list(REGULARIZER_OPTIONS.keys()) + ['None'],
#                         default='l2')
#     parser.add_argument('--regularizer-lambda',
#                         type=float,
#                         help='Lambda to use with kernel regularizer',
#                         default=5e-4)

#     args = parser.parse_args()

#     path_name = args.save_path_name
#     n_epochs = args.n_epochs
#     beta = args.beta # Search grid over this
#     image_shape = args.image_shape
#     latent_dim = args.latent_dim # Search grid over this
#     n_examples_to_generate = args.n_examples_to_generate
#     save_weights_frequency = args.save_weights_frequency
#     kernel_regularizer_type = args.kernel_regularizer
#     lambd = args.regularizer_lambda
#     init_weights_dir = args.init_weights_directory

#     regularizer = REGULARIZER_OPTIONS[kernel_regularizer_type](l=lambd)

#     cur_dir = os.getcwd()
#     save_path = os.path.join(cur_dir, path_name)

#     # Set up tensorboard
#     logdir = os.path.join(save_path, 'logs')
#     file_writer = tf.contrib.summary.create_file_writer(logdir)

#     try:
#         with gzip.open(args.train_data_directory, 'rb') as f:
#             data = pickle.load(f)
#     except:
#         with open(args.train_data_directory, 'rb') as f:
#             data = pickle.load(f)
#     images = data['pixels']
#     try:
#         with gzip.open(args.eval_data_directory, 'rb') as f:
#             data = pickle.load(f)
#     except:
#         with open(args.eval_data_directory, 'rb') as f:
#             data = pickle.load(f)
#     eval_images = data['pixels']
#     print(f'EVALUATING ON {eval_images.shape[0]} IMAGES')

#     # Shuffle data by the same permutation
#     perm = np.random.permutation(images.shape[0])
#     images = images[perm]

#     # Model creation
#     vae = VAE(
#         image_shape=image_shape,
#         latent_dim=latent_dim,
#         kernel_regularizer=regularizer
#     )
#     optimizer = tf.keras.optimizers.Adam(1e-4)
#     vae.encoder.summary()
#     vae.decoder.summary()

#     if init_weights_dir:
#         for weights_path in glob.iglob(os.path.join(init_weights_dir, '*.h5')):
#             if 'encoder' in weights_path:
#                 print('=== LOADING ENCODER WEIGHTS ===')
#                 vae.encoder.load_weights(weights_path)
#             elif 'decoder' in weights_path:
#                 print('=== LOADING DECODER WEIGHTS ===')
#                 vae.decoder.load_weights(weights_path)
#     else:
#         train_dataset, test_dataset, train_data, test_data = get_datasets(images)
#         # If no initial weights, do initial training
#         for epoch in range(1, n_epochs + 1):
#             # Training loop
#             start_time = time.time()
#             for train_x in train_dataset:
#                 compute_apply_gradients(vae, train_x, optimizer, beta)
#             end_time = time.time()

#             if epoch % save_weights_frequency == 0:
#                 vae.encoder.save_weights(
#                     os.path.join(save_path,
#                         f'encoder_{latent_dim}_dim_{beta}_beta.h5'))
#                 vae.decoder.save_weights(
#                     os.path.join(save_path,
#                         f'decoder_{latent_dim}_dim_{beta}_beta.h5'))

#             # Test on eval dataset
#             _elbo = tf.keras.metrics.Mean()
#             _recon_loss, _kl_loss = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
#             for test_x in test_dataset:
#                 recon, kl = compute_elbo_loss_split(vae, test_x, beta=beta)
#                 _elbo(recon + kl)
#                 _recon_loss(recon)
#                 _kl_loss(kl)
#             elbo = -(_elbo.result())
#             recon_loss = -(_recon_loss.result())
#             kl_loss = -(_kl_loss.result())
#             print(f'VAE: {i}, Epoch: {epoch}, Test set ELBO: {elbo},',
#                   f'Reconstruction Loss: {recon_loss}, KL loss: {kl_loss}\n',
#                   f'Time elapsed for current epoch {end_time-start_time}')

#             with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
#                 tf.contrib.summary.scalar(f'vae_{i}/elbo', elbo, step=epoch)
#                 tf.contrib.summary.scalar(f'vae_{i}/reconstruction_loss', recon_loss, step=epoch)
#                 tf.contrib.summary.scalar(f'vae_{i}/kl_divergence', kl_loss, step=epoch)

#             # Evaluate qualitatively on some reconstructions
#             reconstructions = vae.reconstruct(eval_images)
#             concat = np.concatenate([
#                 skimage.util.img_as_ubyte(reconstructions),
#                 eval_images
#             ], axis=2)

#             # for i, (r, orig) in enumerate(zip(reconstructions, random_images)):
#             #     concat = np.concatenate([
#             #         skimage.util.img_as_ubyte(r), orig], axis=1)
#                 # img_path = os.path.join(reconstruct_save_path, f'epoch_{epoch}_{i}.png')
#                 # skimage.io.imsave(img_path, concat)

#             # sampled_vectors = tf.random.normal(
#             #     shape=[n_examples_to_generate, latent_dim])
#             # decoded_samples = vae.decode(sampled_vectors, apply_sigmoid=True)
#             with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
#                 tf.contrib.summary.image(
#                         f'vae_{i}/reconstruction',
#                         concat,
#                         step=epoch,
#                         max_images=concat.shape[0])
#                 # tf.contrib.summary.image(
#                 #         f'vae_{i}/samples',
#                 #         decoded_samples,
#                 #         step=epoch)

#         vae.encoder.save_weights(
#             os.path.join(save_path,
#                 f'encoder_{latent_dim}_dim_{beta}_beta_final.h5'))
#         vae.decoder.save_weights(
#             os.path.join(save_path,
#                 f'decoder_{latent_dim}_dim_{beta}_beta_final.h5'))

#     # Go through test data and calculate highest lowest ELBO
#     train_data, test_data = split_data(images)
#     step_size = 5000
#     data_chunks = [
#         test_data[start_idx:start_idx+step_size, ...]
#         for start_idx in range(0, test_data.shape[0], step_size)
#     ]
#     recon_loss = np.concatenate([
#         np.sum(np.square(chunk - vae.reconstruct(chunk)), axis=(1, 2, 3))
#         for chunk in data_chunks
#     ])

#     # Pull new training data that matches the badly reconstructed data the best
#     N = 10  # Number of samples to collect data for
#     n_worst_idxs = np.argpartition(recon_loss, -N)[-N:]
#     n_worst_recons = test_data[n_worst_idxs]

#     new_train_data, new_train_data_idxs = [], []
#     for worst_recon_img in n_worst_recons:
#         pixel_distances = np.sum(np.square(images - worst_recon_img), axis=(1, 2, 3))
#         smallest_pixel_distance_idxs = np.argpartition(pixel_distances, 1000)[:1000]
#         new_train_data_idxs.extend(smallest_pixel_distance_idxs)
#     new_train_data = images[new_train_data_idxs]

#     # TODO: Loop training with this new data, then repeat


if __name__ == '__main__':
    import argparse
    import time
    import os
    import skimage

    REGULARIZER_OPTIONS = {
        'l1': regularizers.l1,
        'l2': regularizers.l2,
        'l1_l2': regularizers.l1_l2,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dim',
                        type=int,
                        help='Latent dimension of the VAE',
                        default=32)
    parser.add_argument('--image-shape',
                        type=lambda x: eval(x),
                        help='(width, height, channels) of the image',
                        default=(32, 32, 3))
    parser.add_argument('--save-path-name',
                        type=str,
                        help='Name to store the VAE weights under',
                        default='vae')
    parser.add_argument('--n-epochs',
                        type=int,
                        help='Number of epochs to train for',
                        default=250)
    parser.add_argument('--beta',
                        type=lambda x: np.array(eval(x)),
                        help='Beta parameter for the VAE',
                        default=[1.0])
    parser.add_argument('--n-examples-to-generate',
                        type=float,
                        help='Number of random reconstructions to generate',
                        default=4)
    parser.add_argument('--save-weights-frequency',
                        type=int,
                        help='Number of epochs between each save',
                        default=5)
    parser.add_argument('--train-data-directory',
                        type=str,
                        help='.pkl file to load the training data from, formatted as a dictionary with a `pixels` key')
    parser.add_argument('--eval-data-directory',
                        type=str,
                        help='.pkl file to load the evaluation data from, formatted as a dictionary with a `pixels` key')

    parser.add_argument('--kernel-regularizer',
                        type=str,
                        help='Kernel regularizer to use for Conv2D and Dense layers',
                        choices=list(REGULARIZER_OPTIONS.keys()) + ['None'],
                        default='l2')
    parser.add_argument('--regularizer-lambda',
                        type=float,
                        help='Lambda to use with kernel regularizer',
                        default=5e-4)
    parser.add_argument('--include-claw-state',
                        type=lambda x: eval(x),
                        help='Whether or not to include extra state information',
                        default=False)

    args = parser.parse_args()

    path_name = args.save_path_name
    n_epochs = args.n_epochs
    betas = args.beta # Search grid over this
    image_shape = args.image_shape
    latent_dim = args.latent_dim # Search grid over this
    n_examples_to_generate = args.n_examples_to_generate
    save_weights_frequency = args.save_weights_frequency
    kernel_regularizer_type = args.kernel_regularizer
    lambd = args.regularizer_lambda
    include_claw_state = args.include_claw_state

    regularizer = REGULARIZER_OPTIONS[kernel_regularizer_type](l=lambd)

    cur_dir = os.getcwd()
    save_path = os.path.join(cur_dir, path_name)

    # reconstruct_save_path = os.path.join(save_path, 'reconstructions')
    # if not os.path.exists(reconstruct_save_path):
    #     os.makedirs(reconstruct_save_path)

    # Set up tensorboard
    logdir = os.path.join(save_path, 'logs')
    file_writer = tf.contrib.summary.create_file_writer(logdir)

    # Model creation
    all_vaes = [
        VAE(image_shape=image_shape,
            latent_dim=latent_dim,
            beta=_beta,
            extra_input_shape=((9, ) if include_claw_state else None))
        for _beta in betas
    ]
    optimizers = [tf.keras.optimizers.Adam(1e-4) for _ in range(len(all_vaes))]
    all_vaes[0].encoder.summary()
    all_vaes[0].decoder.summary()

    # from softlearning.models.state_estimation import get_dumped_pkl_data
    # images, _ = get_dumped_pkl_data(args.data_directory)

    try:
        with gzip.open(args.train_data_directory, 'rb') as f:
            data = pickle.load(f)
    except:
        with open(args.train_data_directory, 'rb') as f:
            data = pickle.load(f)
    images = data['pixels']
    claw_states = None
    if include_claw_state:
        claw_qpos = data['claw_qpos']
        # last_action = data['last_action']
        claw_states = claw_qpos #np.concatenate([claw_qpos, last_action], axis=1)

    try:
        with gzip.open(args.eval_data_directory, 'rb') as f:
            data = pickle.load(f)
    except:
        with open(args.eval_data_directory, 'rb') as f:
            data = pickle.load(f)
    eval_images = data['pixels']
    if include_claw_state:
        eval_claw_qpos = data['claw_qpos']
        # last_action = data['last_action']
        eval_claw_states = eval_claw_qpos #np.concatenate([claw_qpos, last_action], axis=1)
    print(f'EVALUATING ON {eval_images.shape[0]} IMAGES')

    # Shuffle data by the same permutation
    perm = np.random.permutation(images.shape[0])
    images = images[perm]
    claw_states = (claw_states[perm] if claw_states is not None else None)

    train_dataset, test_dataset, _, _ = get_datasets(images, claw_states=claw_states)

    # vae = VAE(
    #     image_shape=image_shape,
    #     latent_dim=latent_dim,
    #     kernel_regularizer=regularizer
    # )

    # vae.encoder.summary()
    # vae.decoder.summary()
    
    for epoch in range(1, n_epochs + 1):
        # Training loop
        for i, vae in enumerate(all_vaes):
            optimizer = optimizers[i]
            beta = betas[i]
            start_time = time.time()
            for train_x in train_dataset:
                compute_apply_gradients(vae, train_x, optimizer, beta)
            end_time = time.time()

            if epoch % save_weights_frequency == 0:
                vae.encoder.save_weights(
                    os.path.join(save_path,
                        f'encoder_{latent_dim}_dim_{beta}_beta.h5'))
                vae.decoder.save_weights(
                    os.path.join(save_path,
                        f'decoder_{latent_dim}_dim_{beta}_beta.h5'))

            # Test on eval dataset
            _elbo = tf.keras.metrics.Mean()
            _recon_loss, _kl_loss = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
            for test_x in test_dataset:
                recon, kl = compute_elbo_loss_split(vae, test_x, beta=beta)
                _elbo(recon + kl)
                _recon_loss(recon)
                _kl_loss(kl)
            elbo = -(_elbo.result())
            recon_loss = -(_recon_loss.result())
            kl_loss = -(_kl_loss.result())
            print(f'VAE: {i}, Epoch: {epoch}, Test set ELBO: {elbo},',
                  f'Reconstruction Loss: {recon_loss}, KL loss: {kl_loss}\n',
                  f'Time elapsed for current epoch {end_time-start_time}')

            with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(f'vae_{i}/elbo', elbo, step=epoch)
                tf.contrib.summary.scalar(f'vae_{i}/reconstruction_loss', recon_loss, step=epoch)
                tf.contrib.summary.scalar(f'vae_{i}/kl_divergence', kl_loss, step=epoch)

            # Evaluate qualitatively on some reconstructions
            # random_images = images[np.random.randint(images.shape[0],
            #                                          size=n_examples_to_generate)]
            # reconstructions = vae.reconstruct(random_images)
            if include_claw_state:
                reconstructions = vae.reconstruct((eval_images, eval_claw_states))
            else:
                reconstructions = vae.reconstruct(eval_images)
            concat = np.concatenate([
                skimage.util.img_as_ubyte(reconstructions),
                # random_images
                eval_images
            ], axis=2)

            # for i, (r, orig) in enumerate(zip(reconstructions, random_images)):
            #     concat = np.concatenate([
            #         skimage.util.img_as_ubyte(r), orig], axis=1)
                # img_path = os.path.join(reconstruct_save_path, f'epoch_{epoch}_{i}.png')
                # skimage.io.imsave(img_path, concat)

            # sampled_vectors = tf.random.normal(
            #     shape=[n_examples_to_generate, latent_dim])
            # decoded_samples = vae.decode(sampled_vectors, apply_sigmoid=True)
            with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.image(
                        f'vae_{i}/reconstruction',
                        concat,
                        step=epoch,
                        max_images=concat.shape[0])
                # tf.contrib.summary.image(
                #         f'vae_{i}/samples',
                #         decoded_samples,
                #         step=epoch)

    vae.encoder.save_weights(
        os.path.join(save_path,
            f'encoder_{latent_dim}_dim_{beta}_beta_final.h5'))
    vae.decoder.save_weights(
        os.path.join(save_path,
            f'decoder_{latent_dim}_dim_{beta}_beta_final.h5'))
