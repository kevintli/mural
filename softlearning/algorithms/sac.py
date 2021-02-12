import os
from collections import OrderedDict
from numbers import Number

import skimage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm
from softlearning.replay_pools.prioritized_experience_replay_pool import (
    PrioritizedExperienceReplayPool
)

tfd = tfp.distributions


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            Q_targets=None,

            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            her_iters=0,
            save_full_state=False,
            save_eval_paths=False,
            per_alpha=1,
            normalize_ext_reward_gamma=1,
            ext_reward_coeff=1,

            rnd_networks=(),
            rnd_lr=1e-4,
            rnd_int_rew_coeff=0,
            rnd_gamma=0.99,

            use_env_intrinsic_reward=False,

            online_vae=True,
            n_vae_train_steps=50,
            save_reconstruction_frequency=1,
            save_observations=False,

            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int', [grad_steps]): Frequency at which target network
                updates occur.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)
        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = (
            Q_targets if Q_targets
            else tuple(tf.keras.models.clone_model(Q) for Q in Qs)
        )

        self._pool = pool
        if isinstance(self._pool, PrioritizedExperienceReplayPool) and \
           self._pool._mode == 'Bellman_Error':
            self._per = True
            self._per_alpha = per_alpha
        else:
            self._per = False

        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._her_iters = her_iters
        self._base_env = training_environment.unwrapped

        self._save_full_state = save_full_state
        self._save_eval_paths = save_eval_paths

        # VAE
        self._preprocessed_Q_inputs = self._Qs[0].preprocessed_inputs_fn
        self._n_preprocessor_evals_per_epoch = 3
        self._n_vae_train_steps_per_epoch = n_vae_train_steps
        self._online_vae = online_vae

        # Track a few eval pixels to evaluate autoencoders at the same data point
        self._fixed_eval_pixels = None
        self._save_reconstruction_frequency = save_reconstruction_frequency
        self._save_observations = save_observations

        self._normalize_ext_reward_gamma = normalize_ext_reward_gamma
        self._ext_reward_coeff = ext_reward_coeff
        self._running_ext_rew_std = 1
        self._rnd_int_rew_coeff = 0

        self._use_env_intrinsic_reward = use_env_intrinsic_reward

        if rnd_networks:
            self._rnd_target = rnd_networks[0]
            self._rnd_predictor = rnd_networks[1]
            self._rnd_lr = rnd_lr
            self._rnd_int_rew_coeff = rnd_int_rew_coeff
            self._rnd_gamma = rnd_gamma
            self._running_int_rew_std = 1
            # self._rnd_gamma = 1
            # self._running_int_rew_std = 1

        self._build()

    def _build(self):
        super(SAC, self)._build()
        self._init_extrinsic_reward()
        self._init_intrinsic_reward()
        self._init_actor_update()
        self._init_critic_update()

        if self._uses_vae:
            self._init_vae_update()
        elif self._uses_rae:
            self._init_rae_update()
        self._init_diagnostics_ops()

    def _policy_inputs(self, observations):
        policy_inputs = flatten_input_structure({
            name: observations[name]
            for name in self._policy.observation_keys
        })
        return policy_inputs

    def _Q_inputs(self, observations, actions):
        Q_observations = {
            name: observations[name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure(
            {**Q_observations, 'actions': actions})
        return Q_inputs

    def _init_extrinsic_reward(self):
        self._unscaled_ext_reward = self._placeholders['rewards']

    def _init_intrinsic_reward(self):
        # === Using RND ===
        if self._rnd_int_rew_coeff:
            self._init_rnd_update()
            self._unscaled_int_reward = tf.clip_by_value(
                self._rnd_errors / self._placeholders['reward']['running_int_rew_std'],
                0, 1000
            )
            self._int_reward = self._rnd_int_rew_coeff * self._unscaled_int_reward
        # === Use environment reward as intrinsic reward ===
        elif self._use_env_intrinsic_reward:
            self._int_reward = self._placeholders['rewards']
        else:
            self._int_reward = 0

    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.all_keys
        })
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].all_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        # if self._rnd_int_rew_coeff:
        #     self._unscaled_int_reward = tf.clip_by_value(
        #         self._rnd_errors / self._placeholders['reward']['running_int_rew_std'],
        #         0, 1000
        #     )
        #     self._int_reward = self._rnd_int_rew_coeff * self._unscaled_int_reward
        # else:
        #     self._int_reward = 0

        self._normalized_ext_reward = (
            self._unscaled_ext_reward / self._placeholders['reward']['running_ext_rew_std'])

        self._ext_reward = self._normalized_ext_reward * self._ext_reward_coeff
        self._total_reward = self._ext_reward + self._int_reward

        Q_target = td_target(
            reward=self._reward_scale * self._total_reward,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)
        return tf.stop_gradient(Q_target)

    @property
    def _uses_vae(self):
        # Variational Autoencoder (VAE)
        return (
            'pixels' in self._policy.preprocessors and
            self._policy.preprocessors['pixels'].__class__.__name__ == 'OnlineVAEPreprocessor')

    @property
    def _uses_rae(self):
        # Regularized Autoencoder (RAE)
        return (
            'pixels' in self._policy.preprocessors and
            self._policy.preprocessors['pixels'].__class__.__name__ == 'RAEPreprocessor')

    def _init_vae_update(self):
        """
        Initializes VAE optimization update
        Creates a `tf.optimizer.minimize` operation, appends to `self._training_ops`.
        """
        vae_log_dir = os.path.join(os.getcwd(), 'vae')
        if not os.path.exists(vae_log_dir):
            os.makedirs(vae_log_dir)

        self._vae_reconstruction_losses = {}
        self._vae_kl_losses = {}
        self._vae_elbos = {}
        self._vae_latent_l2 = {}
        self._vae_latent_mean = {}
        self._vae_latent_logvar = {}
        self._initial_vae_train_ops = []

        policy_preprocessor = self._policy.preprocessors['pixels']
        Q_preprocessor = self._Qs[0].observations_preprocessors['pixels']
        assert Q_preprocessor is self._Qs[1].observations_preprocessors['pixels'], (
            'Preprocessors on the Qs must be the same object.'
        )

        preprocessors = self._preprocessors = (
            {'policy': policy_preprocessor, 'Q': Q_preprocessor}
            if policy_preprocessor is not Q_preprocessor
            else {'shared': policy_preprocessor}
        )
        for name, vae in preprocessors.items():
        # preprocessors = (('policy', policy_vae_preprocessor),
        #                  ('Q', Q_vae_preprocessor))
        # for (name, preprocessor) in preprocessors:
        #     vae = preprocessor.vae
            images = self._placeholders['observations']['pixels']
            z_mean, z_log_var, z, reconstructions = vae(images, include_reconstructions=True)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=reconstructions,
                labels=tf.image.convert_image_dtype(images, tf.float32)
            )
            # z_mean, z_log_var, z = vae.encoder(images)

            reconstruction_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            kl_divergence = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_divergence = -0.5 * tf.reduce_sum(kl_divergence, axis=-1)

            # ELBO = E(p(x|z)) - D_kl(q(z|x) || p(z)) * beta
            elbo = -reconstruction_loss - vae.beta * kl_divergence
            vae_loss = -tf.reduce_mean(elbo)

            # Log VAE metrics
            self._vae_reconstruction_losses[name] = tf.reduce_mean(
                reconstruction_loss)
            self._vae_kl_losses[name] = tf.reduce_mean(kl_divergence)
            self._vae_elbos[name] = tf.reduce_mean(elbo)
            self._vae_latent_l2[name] = tf.reduce_mean(tf.norm(z, axis=-1))
            self._vae_latent_mean[name] = tf.reduce_mean(z_mean)
            self._vae_latent_logvar[name] = tf.reduce_mean(z_log_var)

            vae_optimizer = tf.train.AdamOptimizer(
                learning_rate=self._policy_lr,
                name=f'{name}_vae_optimizer')

            initial_vae_train_op = tf.contrib.layers.optimize_loss(
                loss=vae_loss,
                global_step=self.global_step,
                learning_rate=self._policy_lr,
                optimizer=vae_optimizer,
                variables=vae.trainable_variables,
                increment_global_step=False,
            )
            vae_train_op = vae_optimizer.minimize(
                loss=vae_loss,
                var_list=vae.trainable_variables
            )
            if self._online_vae:
                self._training_ops.update({f'{name}_vae_train_op': vae_train_op})
            else:
                self._initial_vae_train_ops.append(initial_vae_train_op)

    def _init_rae_update(self):
        """
        Initializes RAE optimization update
        Creates a `tf.optimizer.minimize` operation, appends to `self._training_ops`.
        """
        assert self._uses_rae
        rae_log_dir = os.path.join(os.getcwd(), 'rae')
        if not os.path.exists(rae_log_dir):
            os.makedirs(rae_log_dir)

        self._rae_reconstruction_losses = {}
        self._rae_latent_l2 = {}

        policy_preprocessor = self._policy.preprocessors['pixels']
        # TODO: Do some check to see if the policy AE is the same as the Q
        Q_preprocessor = self._Qs[0].observations_preprocessors['pixels']
        assert Q_preprocessor is self._Qs[1].observations_preprocessors['pixels'], (
            'Preprocessors on the Qs must be the same object.'
        )
        preprocessors = self._preprocessors = (
            {'policy': policy_preprocessor, 'Q': Q_preprocessor}
            if policy_preprocessor is not Q_preprocessor
            else {'shared': policy_preprocessor}
        )
        for name, rae in preprocessors.items():
            images = self._placeholders['observations']['pixels']
            z, reconstructions = rae(images, include_reconstructions=True)

            # L_RAE = ||true - reconstructions||^2 + ||z||^2 + ||theta||^2
            # Sum over individual pixel MSE over the entire image
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                tf.square(
                    tf.image.convert_image_dtype(images, tf.float32) - reconstructions
                ), axis=[1, 2, 3]))
            latent_l2_norm_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z), axis=-1))
            # latent_l2_norm_loss = tf.reduce_mean(tf.norm(z, axis=-1))
            rae_loss = reconstruction_loss + latent_l2_norm_loss

            self._rae_reconstruction_losses[name] = reconstruction_loss
            self._rae_latent_l2[name] = latent_l2_norm_loss

            rae_optimizer = tf.train.AdamOptimizer(
                learning_rate=self._policy_lr,  # TODO: Should this be its own parameter?
                name=f'{name}_rae_optimizer')

            rae_train_op = rae_optimizer.minimize(
                loss=rae_loss,
                var_list=rae.trainable_variables
            )
            self._training_ops.update({f'{name}_rae_train_op': rae_train_op})

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].all_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._bellman_errors = tf.reduce_min(tuple(
            tf.math.squared_difference(Q_target, Q_value)
            for Q_value in Q_values), axis=0)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.all_keys
        })
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].all_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_rnd_update(self):
        self._placeholders['reward'].update({
            'running_int_rew_std': tf.compat.v1.placeholder(
                tf.float32, shape=(), name='running_int_rew_std')
        })
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.all_keys
        })

        targets = tf.stop_gradient(self._rnd_target(policy_inputs))
        predictions = self._rnd_predictor(policy_inputs)

        self._rnd_errors = tf.expand_dims(tf.reduce_mean(
            tf.math.squared_difference(targets, predictions), axis=-1), 1)
        self._rnd_loss = tf.reduce_mean(self._rnd_errors)
        self._rnd_error_std = tf.math.reduce_std(self._rnd_errors)
        self._rnd_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._rnd_lr,
            name="rnd_optimizer")
        rnd_train_op = self._rnd_optimizer.minimize(
            loss=self._rnd_loss)
        self._training_ops.update({
            'rnd_train_op': rnd_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha)
        ))

        if self._rnd_int_rew_coeff:
            diagnosables['rnd_reward'] = self._int_reward
            diagnosables['rnd_error'] = self._rnd_errors
            diagnosables['running_rnd_reward_std'] = self._placeholders[
                'reward']['running_int_rew_std']

        diagnosables['normalized_ext_reward'] = self._normalized_ext_reward
        diagnosables['ext_reward'] = self._ext_reward

        diagnosables['running_ext_reward_std'] = (
            self._placeholders['reward']['running_ext_rew_std'])
        diagnosables['total_reward'] = self._total_reward
        
        if self._int_reward != 0:
            diagnosables['int_reward'] = self._int_reward

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
            ('max', tf.math.reduce_max),
            ('min', tf.math.reduce_min),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

        if self._uses_vae:
            self._diagnostics_ops.update({
                **{
                    f"vae/{key}-reconstruction_loss": value
                    for key, value in self._vae_reconstruction_losses.items()
                },
                **{
                    f"vae/{key}-kl_loss": value
                    for key, value in self._vae_kl_losses.items()
                },
                **{
                    f"vae/{key}-elbo": value
                    for key, value in self._vae_elbos.items()
                },
                **{
                    f"vae/{key}-latent_l2_norm": value
                    for key, value in self._vae_latent_l2.items()
                },
                **{
                    f"vae/{key}-latent_mean": value
                    for key, value in self._vae_latent_mean.items()
                },
                **{
                    f"vae/{key}-latent_logvar": value
                    for key, value in self._vae_latent_logvar.items()
                },
            })
        elif self._uses_rae:
            self._diagnostics_ops.update({
                **{
                    f"rae/{key}-reconstruction_loss": value
                    for key, value in self._rae_reconstruction_losses.items()
                },
                **{
                    f"rae/{key}-latent_l2_norm": value
                    for key, value in self._rae_latent_l2.items()
                },
            })

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(iteration, batch)

        # ======
        # Debugging feed_dict to see what exactly the preprocessed
        # inputs will be; just do something like
        # `self._session.run(preprocessed(inputs_np))`
        # ======

        # import ipdb; ipdb.set_trace()
        # preprocessed = self._Qs[0].preprocessed_inputs_fn
        # inputs_np = [
        #     batch['actions'],
        #     batch['observations']['claw_qpos'],
        #     batch['observations']['last_action'],
        #     batch['observations']['pixels']
        # ]
        # input_dict = {
        #     input_ph: inputs_np[i]
        #     for i, input_ph in enumerate(preprocessed.input)
        # }

        self._session.run(self._training_ops, feed_dict)

        if self._rnd_int_rew_coeff:
            int_rew_std = np.maximum(
                np.std(self._session.run(self._unscaled_int_reward, feed_dict)), 1e-3)
            self._running_int_rew_std = (
                self._running_int_rew_std * self._rnd_gamma
                + int_rew_std * (1 - self._rnd_gamma))

        if self._normalize_ext_reward_gamma != 1:
            ext_rew_std = np.maximum(
                np.std(self._session.run(self._normalized_ext_reward, feed_dict)), 1e-3)
            self._running_ext_rew_std = (
                self._running_ext_rew_std * self._normalize_ext_reward_gamma
                + ext_rew_std * (1 - self._normalize_ext_reward_gamma))

        if self._her_iters:
            # Q: Is it better to build a large batch and take one grad step, or
            # resample many mini batches and take many grad steps?
            new_batches = {}
            for _ in range(self._her_iters):
                new_batch = self._get_goal_resamp_batch(batch)
                new_feed_dict = self._get_feed_dict(iteration, new_batch)
                self._session.run(self._training_ops, new_feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def get_bellman_error(self, batch):
        feed_dict = self._get_feed_dict(None, batch)

        # TO TRY: weight by bellman error without entropy
        # - sweep over per_alpha

        # Question: why the min over the Q's?
        return self._session.run(self._bellman_errors, feed_dict)

    def _epoch_before_hook(self, *args, **kwargs):
        super()._epoch_before_hook(*args, **kwargs)
        # TODO: Create an option for doing training before/after each
        # epoch rather than completely online.
        if hasattr(self, '_online_vae') and not self._online_vae:
            for i in range(self._n_vae_train_steps_per_epoch):
                vae_dataset = self._training_batch(5000)
                feed_dict = self._get_feed_dict(0, vae_dataset)
                for training_op in self._initial_vae_train_ops:
                    self._session.run(
                        training_op,
                        feed_dict=feed_dict)

    def get_grid_vals(self, bins=24, low=-6, high=6):
        xs = np.linspace(low, high, bins)
        ys = np.linspace(low, high, bins)
        xys = np.meshgrid(xs, ys)
        grid_vals = np.array(xys).transpose(1, 2, 0).reshape(-1, 2)
        return grid_vals

    def discretized_states(self, states, bins=None, low=None, high=None):
        """
        Converts continuous to discrete states.
        
        Params
        - states: A shape (n, 2) batch of continuous observations
        - bins: Number of bins for both x and y coordinates
        - low: Lowest value (inclusive) for continuous x and y
        - high: Highest value (inclusive) for continuous x and y
        """
        if bins is None:
            bins = self.n_bins
        if low is None:
            low = self.obs_space.low[0]
        if high is None:
            high = self.obs_space.high[0]

        bin_size = (high - low) / bins
        shifted_states = states - low
        return np.clip(shifted_states // bin_size, 0, bins - 1).astype(np.int32)

    def discrete_to_counts(self, states, bins=16):
        """
        Returns a shape (bins, bins) grid of visitation counts for a batch of
        discrete states.
        """
        counts = np.zeros((bins, bins))
        indices, freqs = np.unique(states, return_counts=True, axis=0)
        indices = indices.astype(np.int32)
        counts[indices[:,0], indices[:,1]] = freqs
        return counts

    def _epoch_after_hook(self, *args, **kwargs):
        super()._epoch_after_hook(*args, **kwargs)

        # return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # === TOTAL VISITATIONS ===
        all_states = self.sampler.pool.last_n_batch(self.sampler.pool.size)['observations']['xy_observation']
        disc = self.discretized_states(all_states, bins=16, low=-6, high=6)
        counts = self.discrete_to_counts(disc, bins=16).T[::-1]
        im = ax1.imshow(counts, extent=(-6, 6, -6, 6))
        im.set_clim(0, 300)
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f"Total visitations")

        # === EPOCH VISITATIONS + REWARDS ===
        last_epoch_batch = self.sampler.pool.last_n_batch(self._epoch_length)
        epoch_states = last_epoch_batch['observations']['xy_observation']
        epoch_rewards = last_epoch_batch['rewards']

        im = ax2.scatter(epoch_states[:,0], epoch_states[:,1], c=epoch_rewards.squeeze())
        im.set_clim(0, self._training_environment.unwrapped.count_bonus_coeff)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        # Also plot start and end of each trajectory separately
        ax2.scatter(epoch_states[0::100,0], epoch_states[0::100,1], marker='*', s=100, color='m')
        ax2.scatter(epoch_states[99::100,0], epoch_states[99::100,1], marker='x', s=100, color='r')
        ax2.set_ylim(-6, 6)
        ax2.set_xlim(-6, 6)
        ax2.set_aspect("equal")
        ax2.set_title(f"Epoch {self._epoch} visitations")

        # === RND REWARDS ===
        # grid_vals = self.get_grid_vals()
        # qpos_rest = np.array([0.565, 1., 0., 0., 0., 0.,
        #         1., 0., -1., 0., -1., 0., 1., -3.,
        #         -3., 0.75, 1., 0., 0., 0., 0., 0.,
        #         0., 0., 0., 0., 0.])
        # exp_bonuses = self._session.run(self._int_reward, feed_dict={
        #     self._placeholders['observations']['state_observation']: 
        #         np.hstack((grid_vals, np.repeat(qpos_rest[None], len(grid_vals), axis=0))),
        #     self._placeholders['observations']['xy_observation']: grid_vals,
        #     self._placeholders['reward']['running_int_rew_std']:
        #         self._running_int_rew_std,
        # })
        # im = ax3.contourf(grid_vals[:,0].reshape(24, 24), grid_vals[:,1].reshape(24, 24), 
        #     exp_bonuses.reshape(24, 24), extent=(-6, 6, -6, 6), levels=30)
        # plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        # ax3.set_title(f"Epoch {self._epoch} RND bonuses")
        # ax3.set_aspect("equal")

        # === COUNT REWARDS ===
        count_bonuses = self._training_environment.get_grid_count_bonuses()
        im = ax3.imshow(count_bonuses[::-1])
        im.set_clim(0, self._training_environment.unwrapped.count_bonus_coeff)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title(f"Epoch {self._epoch} count bonuses")

        # === SAVE FIGURE ===
        save_dir = './exploration_plots/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, 'epoch_%d.png' % self._epoch))
        plt.cla()
        plt.clf()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        if (self._save_observations and
            np.random.rand() < 1e-4 and
            'pixels' in batch['observations']):
            random_idx = np.random.randint(
                batch['observations']['pixels'].shape[0])
            image = batch['observations']['pixels'][random_idx].copy()
            if image.shape[-1] == 6:
                img_0, img_1 = np.split(
                    image, 2, axis=2)
                image = np.concatenate([img_0, img_1], axis=1)
            image_save_dir = os.path.join(os.getcwd(), 'pixels')
            image_save_path = os.path.join(
                image_save_dir, f'observation_{iteration}_batch.png')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            skimage.io.imsave(image_save_path, image)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }
        feed_dict[self._placeholders['rewards']] = batch['rewards']

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        feed_dict[self._placeholders['reward']['running_ext_rew_std']] = (
            self._running_ext_rew_std)
        if self._rnd_int_rew_coeff:
            feed_dict[self._placeholders['reward']['running_int_rew_std']] = (
                self._running_int_rew_std)

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO: We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)
        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(flatten_input_structure({
                name: batch['observations'][name]
                for name in self._policy.all_keys
            })).items()
        ]))

        # VAE diagnostics
        if self._uses_vae or self._uses_rae:
            random_idxs = np.random.choice(
                feed_dict[self._placeholders['observations']['pixels']].shape[0],
                size=self._n_preprocessor_evals_per_epoch)
            eval_pixels = (
                feed_dict[self._placeholders['observations']['pixels']][random_idxs])

        should_save = self._save_reconstruction_frequency == 0
        if self._uses_vae and should_save:
            for name, vae in self._preprocessors.items():
                z_mean, z_logvar, z = self._session.run(vae.encoder(eval_pixels))
                reconstructions = self._session.run(
                    tf.math.sigmoid(vae.decoder(z)))
                concat = np.concatenate([
                    eval_pixels,
                    skimage.util.img_as_ubyte(reconstructions)
                ], axis=2)
                sampled_z = np.random.normal(
                    size=(self._n_preprocessor_evals_per_epoch, vae.latent_dim))
                decoded_samples = self._session.run(
                    tf.math.sigmoid(vae.decoder(sampled_z))
                )

                save_path = os.path.join(os.getcwd(), 'vae')
                recon_concat = np.vstack(concat)
                skimage.io.imsave(
                    os.path.join(
                        save_path,
                        f'{name}_reconstruction_{iteration}.png'),
                    recon_concat)
                samples_concat = np.vstack(decoded_samples)
                skimage.io.imsave(
                    os.path.join(
                        save_path,
                        f'{name}_sample_{iteration}.png'),
                    samples_concat)
        elif self._uses_rae:
            for name, rae in self._preprocessors.items():
                z, reconstructions = self._session.run(
                    rae(eval_pixels, include_reconstructions=True))

                # Keep a set of pixels to track the latents
                if self._fixed_eval_pixels is None:
                    self._fixed_eval_pixels = eval_pixels
                    self._fixed_eval_latents = np.zeros(z.shape)

                if should_save:
                    concat = np.concatenate([
                        eval_pixels,
                        skimage.util.img_as_ubyte(reconstructions)
                    ], axis=2)
                    save_path = os.path.join(os.getcwd(), 'rae')
                    recon_concat = np.vstack(concat)
                    skimage.io.imsave(
                        os.path.join(
                            save_path,
                            f'{name}_reconstruction_{iteration}.png'),
                        recon_concat)

                z_fixed, reconstructions_fixed = self._session.run(
                    rae(self._fixed_eval_pixels, include_reconstructions=True))

                if should_save:
                    concat_fixed = np.concatenate([
                        self._fixed_eval_pixels,
                        skimage.util.img_as_ubyte(reconstructions_fixed)
                    ], axis=2)
                    recon_concat_fixed = np.vstack(concat_fixed)
                    # TODO: Put this concat and saving logic in utils.py
                    skimage.io.imsave(
                        os.path.join(
                            save_path,
                            f'{name}_fixed_reconstruction_{iteration}.png'),
                        recon_concat_fixed)

                z_diff = np.linalg.norm(z_fixed - self._fixed_eval_latents, axis=1)
                diagnostics.update({
                    f'rae/{name}/tracked-latent-l2-difference-with-prev-mean': np.mean(z_diff),
                    f'rae/{name}/tracked-latent-l2-difference-with-prev-std': np.std(z_diff)
                })
                self._fixed_eval_latents = z_fixed

        if self._save_eval_paths:
            import pickle
            file_name = f'eval_paths_{iteration // self.epoch_length}.pkl'
            with open(os.path.join(os.getcwd(), file_name)) as f:
                pickle.dump(evaluation_paths, f)

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
