import os
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import skimage

from softlearning.models.utils import flatten_input_structure
from .sac import SAC
from softlearning.samplers import rollouts
import math
from softlearning.misc.utils import save_video
from flatten_dict import flatten

tfd = tfp.distributions


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class MultiSAC(SAC):
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
            policies,
            Qs_per_policy,
            Q_targets_per_policy,
            pools,
            samplers,
            # goals=(),
            num_goals,
            plotter=None,

            # hyperparams shared across policies
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
            ext_reward_coeffs=[],
            rnd_networks=(),
            rnd_lr=1e-4,
            rnd_int_rew_coeffs=[],
            rnd_gamma=0.99,
            save_reconstruction_frequency=1,
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
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment

        self._policies = policies
        self._Qs_per_policy = Qs_per_policy
        self._samplers = samplers
        self._pools = pools

        self._num_goals = num_goals
        self._goal_index = 0
        self._epoch_length *= num_goals
        self._n_epochs *= num_goals

        error_msg = 'Mismatch between number of policies, Qs, and samplers'
        assert len(self._policies) == num_goals, error_msg
        assert len(self._Qs_per_policy) == num_goals, error_msg
        assert len(self._samplers) == num_goals, error_msg

        self._Q_targets_per_policy = Q_targets_per_policy
        self._training_ops_per_policy = [{} for _ in range(num_goals)]

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
        self._plotter = plotter

        self._her_iters = her_iters
        self._base_env = training_environment.unwrapped

        self._save_full_state = save_full_state
        self._save_eval_paths = save_eval_paths

        self._n_episodes_elapsed = 0
        self._num_grad_steps_taken_per_policy = [0 for _ in range(self._num_goals)]

        self._normalize_ext_reward_gamma = normalize_ext_reward_gamma
        if ext_reward_coeffs:
            self._ext_reward_coeffs = ext_reward_coeffs
        else:
            self._ext_reward_coeffs = [1 for _ in range(num_goals)]
        self._running_ext_rew_stds = [1 for _ in range(num_goals)]
        self._rnd_int_rew_coeffs = []

        if rnd_networks:
            assert len(rnd_networks) == num_goals
            self._rnd_targets = [rnd_network_pair[0] for rnd_network_pair in rnd_networks]
            self._rnd_predictors = [rnd_network_pair[1] for rnd_network_pair in rnd_networks]
            self._rnd_lr = rnd_lr
            self._rnd_int_rew_coeffs = rnd_int_rew_coeffs

            assert len(self._rnd_int_rew_coeffs) == num_goals
            self._rnd_gamma = rnd_gamma
            self._running_int_rew_stds = [1 for _ in range(num_goals)]
            # self._rnd_gamma = 1
            # self._running_int_rew_std = 1

        self._n_preprocessor_evals_per_epoch = 3
        self._online_vae = True
        self._save_reconstruction_frequency = save_reconstruction_frequency
        self._fixed_eval_pixels = None

        self._build()

    def _build(self):
        super(SAC, self)._build()
        self._init_external_rewards()
        self._init_rnd_updates()
        self._init_actor_updates()
        self._init_critic_updates()
        if self._uses_vae:
            self._init_vae_updates()
        elif self._uses_rae:
            self._init_rae_updates()
        self._init_diagnostics_ops()

    def _init_external_rewards(self):
        self._unscaled_ext_rewards = [
            self._placeholders['rewards'] for _ in range(self._num_goals)]

    def _get_Q_targets(self):
        Q_targets = []

        self._placeholders['reward'].update({
            f'running_ext_rew_std_{i}': tf.compat.v1.placeholder(
                tf.float32, shape=(), name=f'running_ext_rew_std_{i}')
            for i in range(self._num_goals)
        })

        (self._unscaled_int_rewards,
         self._int_rewards,
         self._normalized_ext_rewards,
         self._ext_rewards,
         self._total_rewards) = [], [], [], [], []

        for i, policy in enumerate(self._policies):
            policy_inputs = flatten_input_structure({
                name: self._placeholders['next_observations'][name]
                for name in policy.observation_keys
            })
            next_actions = policy.actions(policy_inputs)
            next_log_pis = policy.log_pis(policy_inputs, next_actions)

            next_Q_observations = {
                name: self._placeholders['next_observations'][name]
                for name in self._Qs_per_policy[i][0].observation_keys
            }
            next_Q_inputs = flatten_input_structure(
                {**next_Q_observations, 'actions': next_actions})
            next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets_per_policy[i])

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_values = min_next_Q - self._alphas[i] * next_log_pis

            terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

            if self._rnd_int_rew_coeffs[i]:
                self._unscaled_int_rewards.append(tf.clip_by_value(
                    self._rnd_errors[i] / self._placeholders['reward'][f'running_int_rew_std_{i}'],
                    0, 1000
                ))
            else:
                self._unscaled_int_rewards.append(0)
            self._int_rewards.append(self._rnd_int_rew_coeffs[i] * self._unscaled_int_rewards[i])

            if self._ext_reward_coeffs[i]:
                self._normalized_ext_rewards.append(
                    self._unscaled_ext_rewards[i] / self._placeholders['reward'][f'running_ext_rew_std_{i}'])
            else:
                self._normalized_ext_rewards.append(0)
            self._ext_rewards.append(self._ext_reward_coeffs[i] * self._normalized_ext_rewards[i])

            self._total_rewards.append(self._ext_rewards[i] + self._int_rewards[i])

            Q_target = td_target(
                reward=self._reward_scale * self._total_rewards[i],
                discount=self._discount,
                next_value=(1 - terminals) * next_values)
            Q_targets.append(tf.stop_gradient(Q_target))

        return Q_targets

    @property
    def _uses_vae(self):
        if self._policies and 'pixels' in self._policies[0].preprocessors:
            preprocessor = self._policies[0].preprocessors['pixels']
            return preprocessor.__class__.__name__ == 'OnlineVAEPreprocessor'
        return False

    @property
    def _uses_rae(self):
        if self._policies and 'pixels' in self._policies[0].preprocessors:
            preprocessor = self._policies[0].preprocessors['pixels']
            return preprocessor.__class__.__name__ == 'RAEPreprocessor'
        return False

    def _init_vae_updates(self):
        """
        Initializes VAE optimization update
        Creates a `tf.optimizer.minimize` operation, appends to `self._training_ops`.
        """
        # Need to loop through all the VAEs for Qs (2 per policy), and the VAEs for
        # the policies
        if not self._uses_vae:
            return

        vae_log_dir = os.path.join(os.getcwd(), 'vae')
        if not os.path.exists(vae_log_dir):
            os.makedirs(vae_log_dir)

        self._vae_optimizers_per_policy = []
        # The metrics will be used in `get_diagnostics` below
        self._vae_metrics_per_policy = []
        self._preprocessors_per_policy = []

        for i in range(self._num_goals):
            vae_metrics = {
                'latent_mean': {},
                'latent_logvar': {},
                'elbo': {},
                'reconstruction_loss': {},
                'kl_divergence': {},
            }
            if self._ext_reward_coeffs[i] != 0:
                # Only add update operations attached to preprocessors that are used
                # w.r.t the extrinsic reward
                policy_preprocessor = self._policies[i].preprocessors['pixels']
                Qs = self._Qs_per_policy[i]
                Q_preprocessor = Qs[0].observations_preprocessors['pixels']
                assert Q_preprocessor is Qs[1].observations_preprocessors['pixels'], (
                    'Preprocessors on the critics must be the same object.'
                )
                # preprocessors = (
                #     (f'policy_{i}', policy_preprocessor),
                #     (f'Q_{i}', Q_preprocessor),
                # )
                preprocessors = (
                    {f'policy_{i}': policy_preprocessor, f'Q_{i}': Q_preprocessor}
                    if policy_preprocessor is not Q_preprocessor
                    else {f'shared_{i}': policy_preprocessor}
                )
                self._preprocessors_per_policy.append(preprocessors)

                for name, vae in preprocessors.items():
                    # vae = preprocessor.vae
                    images = self._placeholders['observations']['pixels']
                    z_mean, z_log_var, z, reconstructions = vae(
                        images, include_reconstructions=True)

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
                    vae_metrics['latent_mean'][name] = tf.reduce_mean(z_mean)
                    vae_metrics['latent_logvar'][name] = tf.reduce_mean(z_log_var)
                    vae_metrics['elbo'][name] = tf.reduce_mean(elbo)
                    vae_metrics['reconstruction_loss'][name] = tf.reduce_mean(
                        reconstruction_loss)
                    vae_metrics['kl_divergence'][name] = tf.reduce_mean(kl_divergence)

                    vae_loss = -tf.reduce_mean(elbo)

                    vae_optimizer = tf.train.AdamOptimizer(
                        learning_rate=self._policy_lr,
                        name=f'{name}_vae_optimizer')

                    vae_train_op = vae_optimizer.minimize(
                        loss=vae_loss,
                        var_list=vae.trainable_variables
                    )
                    self._training_ops_per_policy[i].update({
                        f'{name}_vae_train_op': vae_train_op})
            # Add the metrics computed above
            self._vae_metrics_per_policy.append(vae_metrics)

    # TODO: Clean up these two updates, lots of redundancy
    def _init_rae_updates(self):
        """
        Initializes RAE optimization update
        Creates a `tf.optimizer.minimize` operation, appends to `self._training_ops`.
        """
        if not self._uses_rae:
            return

        rae_log_dir = os.path.join(os.getcwd(), 'rae')
        if not os.path.exists(rae_log_dir):
            os.makedirs(rae_log_dir)

        self._rae_optimizers_per_policy = []
        # The metrics will be used in `get_diagnostics` below
        self._rae_metrics_per_policy = []
        self._preprocessors_per_policy = []

        for i in range(self._num_goals):
            rae_metrics = {
                'reconstruction_loss': {},
                'latent_l2_norm': {},
            }
            if self._ext_reward_coeffs[i] != 0:
                # Only add update operations attached to preprocessors that are used
                # w.r.t the extrinsic reward
                policy_preprocessor = self._policies[i].preprocessors['pixels']
                Qs = self._Qs_per_policy[i]
                Q_preprocessor = Qs[0].observations_preprocessors['pixels']
                assert Q_preprocessor is Qs[1].observations_preprocessors['pixels'], (
                    'Preprocessors on the critics must be the same object.'
                )
                # preprocessors = (
                #     (f'policy_{i}', policy_preprocessor),
                #     (f'Q_{i}', Q_preprocessor),
                # )
                preprocessors = (
                    {f'policy_{i}': policy_preprocessor, f'Q_{i}': Q_preprocessor}
                    if policy_preprocessor is not Q_preprocessor
                    else {f'shared_{i}': policy_preprocessor}
                )
                self._preprocessors_per_policy.append(preprocessors)

                for name, rae in preprocessors.items():
                    # rae = preprocessor.rae

                    images = self._placeholders['observations']['pixels']
                    z = rae.encoder(images)
                    reconstructions = rae.decoder(z)

                    # L_RAE = ||true - reconstructions||^2 + ||z||^2 + ||theta||^2
                    # Sum over individual pixel MSE over the entire image
                    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                        tf.square(
                            tf.image.convert_image_dtype(images, tf.float32) - reconstructions
                        ), axis=[1, 2, 3]))
                    latent_l2_norm_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z), axis=-1))
                    # latent_l2_norm_loss = tf.reduce_mean(tf.norm(z, axis=-1))
                    rae_loss = reconstruction_loss + latent_l2_norm_loss

                    rae_metrics['reconstruction_loss'][name] = reconstruction_loss
                    rae_metrics['latent_l2_norm'][name] = latent_l2_norm_loss

                    rae_optimizer = tf.train.AdamOptimizer(
                        learning_rate=self._policy_lr,
                        name=f'{name}_rae_optimizer')

                    rae_train_op = rae_optimizer.minimize(
                        loss=rae_loss,
                        var_list=rae.trainable_variables
                    )
                    self._training_ops_per_policy[i].update({
                        f'{name}_rae_train_op': rae_train_op})
            # Add the metrics computed above
            self._rae_metrics_per_policy.append(rae_metrics)

    def _init_critic_updates(self):
        """Create minimization operation for critics' Q-functions.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._get_Q_targets()
        assert len(Q_targets) == len(self._policies)
        for Q_target in Q_targets:
            assert Q_target.shape.as_list() == [None, 1]

        self._Q_optimizers_per_policy = []
        self._Q_values_per_policy = []
        self._Q_losses_per_policy = []

        for i, Qs in enumerate(self._Qs_per_policy):
            Q_observations = {
                name: self._placeholders['observations'][name]
                for name in Qs[0].observation_keys
            }
            Q_inputs = flatten_input_structure({
                **Q_observations, 'actions': self._placeholders['actions']})

            Q_values = tuple(Q(Q_inputs) for Q in Qs)
            self._Q_values_per_policy.append(Q_values)

            Q_losses = tuple(
                tf.compat.v1.losses.mean_squared_error(
                    labels=Q_targets[i], predictions=Q_value, weights=0.5)
                for Q_value in Q_values)
            self._Q_losses_per_policy.append(Q_losses)

            # self._bellman_errors.append(tf.reduce_min(tuple(
            #     tf.math.squared_difference(Q_target, Q_value)
            #     for Q_value in Q_values), axis=0))

            Q_optimizers = tuple(
                tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self._Q_lr,
                    name='{}_{}_optimizer_{}'.format(i, Q._name, j)
                ) for j, Q in enumerate(Qs))
            self._Q_optimizers_per_policy.append(Q_optimizers)

            Q_training_ops = tuple(
                Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
                for i, (Q, Q_loss, Q_optimizer)
                in enumerate(zip(Qs, Q_losses, Q_optimizers)))

            self._training_ops_per_policy[i].update({f'Q_{i}': tf.group(Q_training_ops)})

    def _init_actor_updates(self):
        """Create minimization operations for policies and entropies.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        self._log_alphas = []
        self._alpha_optimizers = []
        self._alphas = []
        self._policy_optimizers = []
        self._policy_losses = []

        for i, policy in enumerate(self._policies):
            policy_inputs = flatten_input_structure({
                name: self._placeholders['observations'][name]
                for name in policy.observation_keys
            })
            actions = policy.actions(policy_inputs)
            log_pis = policy.log_pis(policy_inputs, actions)

            assert log_pis.shape.as_list() == [None, 1]

            log_alpha = tf.compat.v1.get_variable(
                f'log_alpha_{i}',
                dtype=tf.float32,
                initializer=0.0)
            alpha = tf.exp(log_alpha)
            self._log_alphas.append(log_alpha)
            self._alphas.append(alpha)

            if isinstance(self._target_entropy, Number):
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

                alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                    self._policy_lr, name=f'alpha_optimizer_{i}')
                self._alpha_optimizers.append(alpha_optimizer)
                alpha_train_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
                self._training_ops_per_policy[i].update({
                    f'temperature_alpha_{i}': alpha_train_op
                })

            if self._action_prior == 'normal':
                policy_prior = tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self._action_shape),
                    scale_diag=tf.ones(self._action_shape))
                policy_prior_log_probs = policy_prior.log_prob(actions)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            Q_observations = {
                name: self._placeholders['observations'][name]
                for name in self._Qs_per_policy[i][0].observation_keys
            }
            Q_inputs = flatten_input_structure({
                **Q_observations, 'actions': actions})
            Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs_per_policy[i])
            min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

            if self._reparameterize:
                policy_kl_losses = (
                    alpha * log_pis
                    - min_Q_log_target
                    - policy_prior_log_probs)
            else:
                raise NotImplementedError

            assert policy_kl_losses.shape.as_list() == [None, 1]

            self._policy_losses.append(policy_kl_losses)
            policy_loss = tf.reduce_mean(policy_kl_losses)

            policy_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._policy_lr,
                name=f"policy_optimizer_{i}")

            self._policy_optimizers.append(policy_optimizer)

            policy_train_op = policy_optimizer.minimize(
                loss=policy_loss,
                var_list=policy.trainable_variables)

            self._training_ops_per_policy[i].update({f'policy_train_op_{i}': policy_train_op})

    def _init_rnd_updates(self):
        (self._rnd_errors,
         self._rnd_losses,
         self._rnd_error_stds,
         self._rnd_optimizers) = [], [], [], []
        for i in range(self._num_goals):
            self._placeholders['reward'].update({
                f'running_int_rew_std_{i}': tf.compat.v1.placeholder(
                    tf.float32, shape=(), name=f'running_int_rew_std_{i}')
            })
            policy_inputs = flatten_input_structure({
                name: self._placeholders['observations'][name]
                for name in self._policies[i].observation_keys
            })

            targets = tf.stop_gradient(self._rnd_targets[i](policy_inputs))
            predictions = self._rnd_predictors[i](policy_inputs)

            self._rnd_errors.append(tf.expand_dims(tf.reduce_mean(
                tf.math.squared_difference(targets, predictions), axis=-1), 1))
            self._rnd_losses.append(tf.reduce_mean(self._rnd_errors[i]))
            self._rnd_error_stds.append(tf.math.reduce_std(self._rnd_errors[i]))
            self._rnd_optimizers.append(tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._rnd_lr,
                name=f"rnd_optimizer_{i}"))
            rnd_train_op = self._rnd_optimizers[i].minimize(
                loss=self._rnd_losses[i])
            self._training_ops_per_policy[i].update(
                {f'rnd_train_op_{i}': rnd_train_op}
            )

    def _init_diagnostics_ops(self):
        diagnosables_per_goal = [
            OrderedDict((
                (f'Q_value_{i}', self._Q_values_per_policy[i]),
                (f'Q_loss_{i}', self._Q_losses_per_policy[i]),
                (f'policy_loss_{i}', self._policy_losses[i]),
                (f'alpha_{i}', self._alphas[i])
            ))
            for i in range(self._num_goals)
        ]

        for i in range(self._num_goals):
            # Only record the intrinsic/extrinsic reward diagnostics if
            # the reward is actually used (i.e. the reward coeff is not 0)
            if self._rnd_int_rew_coeffs[i]:
                diagnosables_per_goal[i][f'rnd_reward_{i}'] = self._int_rewards[i]
                diagnosables_per_goal[i][f'rnd_error_{i}'] = self._rnd_errors[i]
                diagnosables_per_goal[i][f'running_rnd_reward_std_{i}'] = (
                    self._placeholders['reward'][f'running_int_rew_std_{i}'])

            if self._ext_reward_coeffs[i]:
                diagnosables_per_goal[i][f'ext_reward_{i}'] = self._ext_rewards[i]
                diagnosables_per_goal[i][f'normalized_ext_reward_{i}'] = (
                    self._normalized_ext_rewards[i])
                diagnosables_per_goal[i][f'unnormalized_ext_reward_{i}'] = (
                    self._unscaled_ext_rewards[i])

            diagnosables_per_goal[i][f'running_ext_reward_std_{i}'] = (
                self._placeholders['reward'][f'running_ext_rew_std_{i}'])
            diagnosables_per_goal[i][f'total_reward_{i}'] = self._total_rewards[i]

        if self._uses_vae:
            for i in range(self._num_goals):
                vae_metrics = self._vae_metrics_per_policy[i]
                for metric, logs in vae_metrics.items():
                    for name, log in logs.items():
                        diagnostic_key = f'vae/{name}/{metric}'
                        diagnosables_per_goal[i][diagnostic_key] = log
        elif self._uses_rae:
            for i in range(self._num_goals):
                rae_metrics = self._rae_metrics_per_policy[i]
                for metric, logs in rae_metrics.items():
                    for name, log in logs.items():
                        diagnostic_key = f'rae/{name}/{metric}'
                        diagnosables_per_goal[i][diagnostic_key] = log

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
            ('max', tf.math.reduce_max),
            ('min', tf.math.reduce_min),
        ))

        self._diagnostics_ops_per_goal = [
            OrderedDict([
                (f'{key}-{metric_name}', metric_fn(values))
                for key, values in diagnosables.items()
                for metric_name, metric_fn in diagnostic_metrics.items()
            ])
            for diagnosables in diagnosables_per_goal
        ]

    def _training_batch(self, batch_size=None):
        return self._samplers[self._goal_index].random_batch(batch_size)

    def _evaluation_batches(self, batch_size=None):
        return [self._samplers[i].random_batch(batch_size) for i in range(self._num_goals)]

    def _update_target(self, i, tau=None):
        """ Update target networks for policy i. """
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs_per_policy[i], self._Q_targets_per_policy[i]):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _epoch_before_hook(self, *args, **kwargs):
        super(SAC, self)._epoch_before_hook(*args, **kwargs)

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops_per_policy[self._goal_index], feed_dict)

        if self._rnd_int_rew_coeffs[self._goal_index]:
            int_rew_std = np.maximum(np.std(self._session.run(
                self._unscaled_int_rewards[self._goal_index], feed_dict)), 1e-3)
            self._running_int_rew_stds[
                self._goal_index] = self._running_int_rew_stds[self._goal_index] * self._rnd_gamma + int_rew_std * (1-self._rnd_gamma)

        if self._normalize_ext_reward_gamma != 1 and self._ext_reward_coeffs[self._goal_index]:
            ext_rew_std = np.maximum(np.std(self._session.run(
                self._normalized_ext_rewards[self._goal_index], feed_dict)), 1e-3)
            self._running_ext_rew_stds[
                self._goal_index] = self._running_ext_rew_stds[self._goal_index] * self._normalize_ext_reward_gamma + \
                ext_rew_std * (1-self._normalize_ext_reward_gamma)

        self._num_grad_steps_taken_per_policy[self._goal_index] += 1

        if self._num_grad_steps_taken_per_policy[self._goal_index] % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(self._goal_index)

    def _get_feed_dict(self, iteration, batch):
        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        # if np.random.rand() < 1e-4 and 'pixels' in batch['observations']:
        #     import os
        #     from skimage import io
        #     random_idx = np.random.randint(
        #         batch['observations']['pixels'].shape[0])
        #     image_save_dir = os.path.join(os.getcwd(), 'pixels')
        #     image_save_path = os.path.join(
        #         image_save_dir, f'observation_{iteration}_batch.png')
        #     if not os.path.exists(image_save_dir):
        #         os.makedirs(image_save_dir)
        #     io.imsave(image_save_path,
        #               batch['observations']['pixels'][random_idx].copy())

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        ext_rew_std_ph = self._placeholders['reward'][f'running_ext_rew_std_{self._goal_index}']
        feed_dict[ext_rew_std_ph] = self._running_ext_rew_stds[self._goal_index]
        if self._rnd_int_rew_coeffs[self._goal_index]:
            int_rew_std_ph = self._placeholders['reward'][f'running_int_rew_std_{self._goal_index}']
            feed_dict[int_rew_std_ph] = self._running_int_rew_stds[self._goal_index]

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batches,
                        training_paths_per_policy,
                        evaluation_paths_per_policy):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        goal_index = self._goal_index
        diagnostics = {}

        for i in range(self._num_goals):
            self._goal_index = i
            feed_dict = self._get_feed_dict(iteration, batches[i])
            diagnostics.update(
                self._session.run({**self._diagnostics_ops_per_goal[i]}, feed_dict))
            diagnostics.update(OrderedDict([
                (f'policy_{i}/{key}', value)
                for key, value in self._policies[i].get_diagnostics(
                    flatten_input_structure({
                        name: batches[i]['observations'][name]
                        for name in self._policies[i].observation_keys})
                ).items()
            ]))
        self._goal_index = goal_index
        should_save = iteration % self._save_reconstruction_frequency == 0

        # Generate random pixels to evaluate the preprocessors
        if 'pixels' in self._placeholders['observations']:
            random_idxs = np.random.choice(
                feed_dict[self._placeholders['observations']['pixels']].shape[0],
                size=self._n_preprocessor_evals_per_epoch)
            eval_pixels = (
                feed_dict[self._placeholders['observations']['pixels']][random_idxs])
        else:
            eval_pixels = None

        if self._uses_vae and should_save:
            assert eval_pixels
            for i, preprocessors in enumerate(self._preprocessors_per_policy):
                if self._ext_reward_coeffs[i] == 0:
                    continue

                for name, vae in preprocessors.items():
                    z_mean, z_logvar, z = self._session.run(vae.encoder(eval_pixels))
                    reconstructions = self._session.run(
                        tf.math.sigmoid(vae.decoder(z)))
                    concat = np.concatenate([
                        eval_pixels,
                        skimage.util.img_as_ubyte(reconstructions)
                    ], axis=2)

                    sampled_z = np.random.normal(
                        size=(eval_pixels.shape[0], vae.latent_dim))
                    decoded_samples = self._session.run(
                        tf.math.sigmoid(vae.decoder.output),
                        feed_dict={vae.decoder.input: sampled_z}
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
            assert eval_pixels
            for i, preprocessors in enumerate(self._preprocessors_per_policy):
                if self._ext_reward_coeffs[i] == 0:
                    continue
                for name, rae in preprocessors.items():
                    z, reconstructions = self._session.run(
                        rae(eval_pixels, include_reconstructions=True))
                    concat = np.concatenate([
                        eval_pixels,
                        skimage.util.img_as_ubyte(reconstructions)
                    ], axis=2)

                    if should_save:
                        save_path = os.path.join(os.getcwd(), 'rae')
                        recon_concat = np.vstack(concat)
                        skimage.io.imsave(
                            os.path.join(
                                save_path,
                                f'{name}_reconstruction_{iteration}.png'),
                            recon_concat)

                    # Track latents
                    if self._fixed_eval_pixels is None:
                        self._fixed_eval_pixels = eval_pixels
                        self._fixed_eval_latents = np.zeros(z.shape)

                    z_fixed, reconstructions_fixed = self._session.run(
                        rae(self._fixed_eval_pixels, include_reconstructions=True))

                    if should_save:
                        concat_fixed = np.concatenate([
                            self._fixed_eval_pixels,
                            skimage.util.img_as_ubyte(reconstructions_fixed)
                        ], axis=2)
                        recon_concat_fixed = np.vstack(concat_fixed)
                        skimage.io.imsave(
                            os.path.join(
                                save_path,
                                f'{name}_fixed_reconstruction_{iteration}.png'),
                            recon_concat_fixed)

                    z_diff = np.linalg.norm(z_fixed - self._fixed_eval_latents, axis=1)
                    diagnostics.update({
                        f'rae/{name}/tracked-latent-l2-difference-with-prev-mean': np.mean(z_diff),
                        f'rae/{name}/tracked-latent-l2-difference-with-prev-std': np.std(z_diff),
                    })
                    # Save the previous latents to compare in the next epoch
                    self._fixed_eval_latents = z_fixed

        if self._save_eval_paths:
            import pickle
            file_name = f'eval_paths_{iteration // self.epoch_length}.pkl'
            with open(os.path.join(os.getcwd(), file_name)) as f:
                pickle.dump(evaluation_paths_per_policy, f)

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            **{
                f'_policy_optimizer_{i}': policy_optimizer
                for i, policy_optimizer in enumerate(self._policy_optimizers)
            },
            **{
                f'_log_alphas_{i}': log_alpha
                for i, log_alpha in enumerate(self._log_alphas)
            },
        }

        Q_optimizer_saveables = [
            {
                f'Q_optimizer_{i}_{j}': Q_optimizer
                for j, Q_optimizer in enumerate(Q_optimizers)
            }
            for i, Q_optimizers in enumerate(self._Q_optimizers_per_policy)
        ]

        for Q_opt_dict in Q_optimizer_saveables:
            saveables.update(Q_opt_dict)

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables

    def _initial_exploration_hook(self, env, initial_exploration_policy, goal_index):
        print("start random exploration")
        if self._n_initial_exploration_steps < 1:
            return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        env.set_goal(goal_index)

        self._samplers[goal_index].initialize(env, initial_exploration_policy, self._pools[goal_index])
        while self._pools[goal_index].size < self._n_initial_exploration_steps:
            self._samplers[goal_index].sample()

    def _init_training(self):
        for i in range(self._num_goals):
            self._update_target(i, tau=1.0)

    def _initialize_samplers(self):
        for i in range(self._num_goals):
            self._samplers[i].initialize(self._training_environment, self._policies[i], self._pools[i])
            self._samplers[i].set_save_training_video_frequency(self._save_training_video_frequency * self._num_goals)
        self._n_episodes_elapsed = sum([self._samplers[i]._n_episodes for i in range(self._num_goals)])

    @property
    def ready_to_train(self):
        return self._samplers[self._goal_index].batch_ready()

    def _do_sampling(self, timestep):
        self._sample_count += 1
        self._samplers[self._goal_index].sample()

    def _set_goal(self, goal_index):
        """ Set goal in env. """
        assert goal_index >= 0 and goal_index < self._num_goals
        # print("setting goal to: ", goal_index, ", n_episodes_elapsed: ", self._n_episodes_elapsed)
        self._goal_index = goal_index
        self._training_environment.set_goal(self._goal_index) # TODO: implement in env

    def _timestep_before_hook(self, *args, **kwargs):
        # Check if an entire trajectory has been completed
        n_episodes_sampled = sum([self._samplers[i]._n_episodes for i in range(self._num_goals)])
        if n_episodes_sampled > self._n_episodes_elapsed:
            self._n_episodes_elapsed = n_episodes_sampled
            new_goal_index = (self._goal_index + 1) % self._num_goals
            self._set_goal(new_goal_index)

    def _train(self):
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        import gtimer as gt
        from itertools import count
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        training_metrics = [0 for _ in range(self._num_goals)]

        if not self._training_started:
            self._init_training()

            for i in range(self._num_goals):
                self._initial_exploration_hook(
                    training_environment, self._initial_exploration_policy, i)

        self._initialize_samplers()
        self._sample_count = 0

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        print("starting_training")
        self._training_before_hook()
        import time

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')
            start_samples = sum([self._samplers[i]._total_samples for i in range(self._num_goals)])
            sample_times = []
            for i in count():
                samples_now = sum([self._samplers[i]._total_samples for i in range(self._num_goals)])
                self._timestep = samples_now - start_samples

                if samples_now >= start_samples + self._epoch_length and self.ready_to_train:
                    break

                t0 = time.time()
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')
                sample_times.append(time.time() - t0)
                t0 = time.time()
                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)
                gt.stamp('train')
                # print("Train time: ", time.time() - t0)

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            # TODO diagnostics per goal
            print("Average Sample Time: ", np.mean(np.array(sample_times)))
            print("Step count", self._sample_count)
            training_paths_per_policy = self._training_paths()
            # self.sampler.get_last_n_paths(
            #     math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths_per_policy = self._evaluation_paths()
            gt.stamp('evaluation_paths')

            training_metrics_per_policy = self._evaluate_rollouts(
                training_paths_per_policy, training_environment)
            gt.stamp('training_metrics')

            if evaluation_paths_per_policy:
                evaluation_metrics_per_policy = self._evaluate_rollouts(
                    evaluation_paths_per_policy, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics_per_policy = [{} for _ in range(self._num_goals)]

            self._epoch_after_hook(training_paths_per_policy)
            gt.stamp('epoch_after_hook')

            t0 = time.time()

            sampler_diagnostics_per_policy = [
                self._samplers[i].get_diagnostics() for i in range(self._num_goals)]

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batches=self._evaluation_batches(),
                training_paths_per_policy=training_paths_per_policy,
                evaluation_paths_per_policy=evaluation_paths_per_policy)

            time_diagnostics = gt.get_times().stamps.itrs

            print("Basic diagnostics: ", time.time() - t0)
            print("Sample count: ", self._sample_count)

            diagnostics.update(OrderedDict((
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            print("Other basic diagnostics: ", time.time() - t0)
            for i, (evaluation_metrics, training_metrics, sampler_diagnostics) in (
                enumerate(zip(evaluation_metrics_per_policy,
                              training_metrics_per_policy,
                              sampler_diagnostics_per_policy))):
                diagnostics.update(OrderedDict((
                    *(
                        (f'evaluation_{i}/{key}', evaluation_metrics[key])
                        for key in sorted(evaluation_metrics.keys())
                    ),
                    *(
                        (f'training_{i}/{key}', training_metrics[key])
                        for key in sorted(training_metrics.keys())
                    ),
                    *(
                        (f'sampler_{i}/{key}', sampler_diagnostics[key])
                        for key in sorted(sampler_diagnostics.keys())
                    ),
                )))

            # if self._eval_render_kwargs and hasattr(
            #         evaluation_environment, 'render_rollouts'):
            #     # TODO: Make this consistent such that there's no
            #     # need for the hasattr check.
            #     training_environment.render_rollouts(evaluation_paths)

            yield diagnostics
            print("Diagnostic time: ",  time.time() - t0)

        for i in range(self._num_goals):
            self._samplers[i].terminate()

        self._training_after_hook()

        del evaluation_paths_per_policy

        yield {'done': True, **diagnostics}

    def _training_paths(self):
        """ Override to interleave training videos between policy rollouts. """
        num_paths = math.ceil(
            (self._epoch_length / self._num_goals) / self._samplers[self._goal_index]._max_path_length)
        paths_per_policy = [
            self._samplers[i].get_last_n_paths(num_paths)
            for i in range(self._num_goals)
        ]

        if self._save_training_video_frequency:
            fps = 1 // getattr(self._training_environment, 'dt', 1/30)
            i = 0
            for rollout_num in reversed(range(len(paths_per_policy[0]))):
                if i % self._save_training_video_frequency == 0:
                    video_frames = []
                    for goal_index in range(self._num_goals):
                        video_frames.extend(paths_per_policy[goal_index][rollout_num].pop('images'))
                    video_file_name = f'training_path_{self._epoch}_{i}.mp4'
                    video_file_path = os.path.join(
                        os.getcwd(), 'videos', video_file_name)
                    save_video(video_frames, video_file_path, fps=fps)
                    i += 1

        return paths_per_policy

    def _evaluation_paths(self):
        if self._eval_n_episodes < 1:
            return ()

        should_save_video = (
            self._video_save_frequency > 0
            and self._epoch % self._video_save_frequency == 0)

        paths = []
        for goal_index in range(self._num_goals):
            with self._policies[goal_index].set_deterministic(self._eval_deterministic):
                self._evaluation_environment.set_goal(goal_index)
                paths.append(
                    rollouts(
                        self._eval_n_episodes,
                        self._evaluation_environment,
                        self._policies[goal_index],
                        self._samplers[goal_index]._max_path_length,
                        render_kwargs=(self._eval_render_kwargs
                                       if should_save_video else {})
                    )
                )

        # TODO: interleave videos from different policies
        if should_save_video:
            fps = 1 // getattr(self._evaluation_environment, 'dt', 1/30)
            for rollout_num in range(len(paths[0])):
                video_frames = []
                for goal_index in range(self._num_goals):
                    video_frames.append(paths[goal_index][rollout_num].pop('images'))
                video_frames = np.concatenate(video_frames)
                video_file_name = f'evaluation_path_{self._epoch}_{rollout_num}.mp4'
                video_file_path = os.path.join(
                    os.getcwd(), 'videos', video_file_name)
                save_video(video_frames, video_file_path, fps=fps)
        return paths

    def _evaluate_rollouts(self, episodes_per_policy, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics_per_policy = []
        for i, episodes in enumerate(episodes_per_policy):
            episodes_rewards = [episode['rewards'] for episode in episodes]
            episodes_reward = [np.sum(episode_rewards)
                               for episode_rewards in episodes_rewards]
            episodes_length = [episode_rewards.shape[0]
                               for episode_rewards in episodes_rewards]

            diagnostics = OrderedDict((
                ('episode-reward-mean', np.mean(episodes_reward)),
                ('episode-reward-min', np.min(episodes_reward)),
                ('episode-reward-max', np.max(episodes_reward)),
                ('episode-reward-std', np.std(episodes_reward)),
                ('episode-length-mean', np.mean(episodes_length)),
                ('episode-length-min', np.min(episodes_length)),
                ('episode-length-max', np.max(episodes_length)),
                ('episode-length-std', np.std(episodes_length)),
            ))

            env_infos = env.get_path_infos(episodes)
            for key, value in env_infos.items():
                diagnostics[f'env_infos/{key}'] = value

            diagnostics_per_policy.append(diagnostics)

        return diagnostics_per_policy
