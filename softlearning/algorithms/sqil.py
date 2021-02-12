import numpy as np
import tensorflow as tf

from .sac import SAC
from softlearning.models.utils import flatten_input_structure
from flatten_dict import flatten


class SQIL(SAC):
    def __init__(
        self,
        goal_transitions,
        goal_negative_ratio=1.0,
        lambda_samp=1.0,
        **kwargs,
    ):
        self._goal_transitions = goal_transitions

        self._goal_negative_ratio = goal_negative_ratio
        self._lambda_samp = lambda_samp

        self._num_goal_transitions = goal_transitions['actions'].shape[0]

        # Fill goal transitions with reward 1 terminal states
        self._goal_transitions['rewards'] = np.ones(
            (self._num_goal_transitions, 1),
            dtype=np.float32)
        self._goal_transitions['terminals'] = np.full(
            (self._num_goal_transitions, 1),
            fill_value=True,
            dtype=np.bool)

        self._goal_transitions_flat = flatten(self._goal_transitions)

        super(SQIL, self).__init__(**kwargs)

    def _build(self):
        super(SQIL, self)._build()
        # self._init_goal_placeholders()

    def _init_goal_placeholders(self):
        self._placeholders.update({
            'goal_observations': tf.compat.v1.placeholder(
                tf.float32,
                shape=self._placeholders['observations'].shape,
                name='goal_observations'
            ),
            'goal_next_observations': tf.compat.v1.placeholder(
                tf.float32,
                shape=self._placeholders['next_observations'].shape,
                name='goal_next_observations'
            ),
            'goal_actions': tf.compat.v1.placeholder(
                tf.float32,
                shape=self._placeholders['actions'].shape,
                name='goal_actions'
            ),
            'goal_terminals': tf.compat.v1.placeholder(
                tf.bool,
                shape=self._placeholders['terminals'].shape,
                name='goal_terminals'
            ),
            'goal_rewards': tf.compat.v1.placeholder(
                tf.float32,
                shape=self._placeholders['rewards'].shape,
                name='goal_rewards'
            ),
        })

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

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        # TODO: Add in filtering based on the reward
        # samp_Q_target = self._get_Q_target()
        # assert samp_Q_target.shape.as_list() == [None, 1]
        # samp_Q_inputs = self._Q_inputs(
        #     observations=self._placeholders['observations'],
        #     actions=self._placeholders['actions'])
        # samp_Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)
        # samp_Q_losses = tuple(
        #     tf.compat.v1.losses.mean_squared_error(
        #         labels=samp_Q_target, predictions=samp_Q_value, weights=0.5)
        #     for samp_Q_value in samp_Q_values)

        # goal_Q_target = self._placeholders['goal_reward']
        # assert goal_Q_target.shape.as_list() == [None, 1]
        # goal_Q_inputs = self._Q_inputs(
        #     observations=self._placeholders['goal_observations'],
        #     actions=self._placeholders['goal_actions'])
        # goal_Q_values = tuple(Q(goal_Q_values) for Q in self._Qs)
        # goal_Q_losses = tuple(
        #     tf.compat.v1.losses.mean_squared_error(
        #         labels=goal_Q_target, predictions=goal_Q_value, weights=0.5)
        #     for goal_Q_value in goal_Q_values)

        # # See Algorithm 1 in SQIL
        # Q_losses = self._Q_losses = tuple(
        #     tf.add(goal_Q_loss, tf.mul(self._lambda_samp, samp_Q_loss))
        #     for goal_Q_loss, samp_Q_loss in zip(samp_Q_losses, goal_Q_losses))

        samp_Q_target = self._get_Q_target()
        assert samp_Q_target.shape.as_list() == [None, 1]
        samp_Q_inputs = self._Q_inputs(
            observations=self._placeholders['observations'],
            actions=self._placeholders['actions'])
        samp_Q_values = self._Q_values = tuple(Q(samp_Q_inputs) for Q in self._Qs)
        samp_Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=samp_Q_target, predictions=samp_Q_value, weights=0.5)
            for samp_Q_value in samp_Q_values)
        Q_losses = self._Q_losses = samp_Q_losses

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

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        samp_feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }
        samp_feed_dict[self._placeholders['rewards']] = np.zeros(batch['rewards'].shape)

        # TODO: Allow for different ratio of data
        goal_feed_dict = self._get_goal_feed_dict(
            batch_size=batch_flat[next(iter(batch_flat))].shape[0])

        feed_dict = {
            key: np.concatenate([samp_feed_dict[key], goal_feed_dict[key]], axis=0)
            for key in samp_feed_dict
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        feed_dict[self._placeholders['reward']['running_ext_rew_std']] = (
            self._running_ext_rew_std)
        if self._rnd_int_rew_coeff:
            feed_dict[self._placeholders['reward']['running_int_rew_std']] = (
                self._running_int_rew_std)

        return feed_dict

    def _get_goal_feed_dict(self, batch_size):
        rand_idxs = np.random.randint(
            self._num_goal_transitions,
            size=batch_size)

        placeholders_flat = flatten(self._placeholders)

        goal_feed_dict = {
            placeholders_flat[key]: self._goal_transitions_flat[key][rand_idxs]
            for key in placeholders_flat
            if key in self._goal_transitions_flat.keys()
        }
        goal_feed_dict[self._placeholders['reward']['running_ext_rew_std']] = (
            self._running_ext_rew_std)
        if self._rnd_int_rew_coeff:
            goal_feed_dict[self._placeholders['reward']['running_int_rew_std']] = (
                self._running_int_rew_std)

        return goal_feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SQIL, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        samp_obs = batch['observations']
        batch_size = samp_obs[next(iter(samp_obs))].shape[0]

        samp_Q_values, samp_Q_losses = self._session.run(
            (self._Q_values, self._Q_losses),
            feed_dict=super(SQIL, self)._get_feed_dict(iteration, batch))
        goal_Q_values, goal_Q_losses = self._session.run(
            (self._Q_values, self._Q_losses),
            feed_dict=self._get_goal_feed_dict(batch_size))

        # TODO: Add validation?
        diagnostics.update({
            'sqil/sample_Q_values': np.mean(samp_Q_values),
            'sqil/goal_Q_values': np.mean(goal_Q_values),
            'sqil/sample_Q_losses': np.mean(samp_Q_losses),
            'sqil/goal_Q_losses': np.mean(goal_Q_losses),
        })

        return diagnostics
