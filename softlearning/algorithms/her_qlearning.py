import numpy as np
import tensorflow as tf

from .sac import SAC
from softlearning.models.utils import flatten_input_structure
from flatten_dict import flatten


class HERQLearning(SAC):
    def __init__(
        self,
        replace_original_reward=True,
        **kwargs,
    ):
        self._replace_original_reward = replace_original_reward
        super(HERQLearning, self).__init__(**kwargs)

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

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        orig_feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }
        if self._replace_original_reward:
            orig_feed_dict[self._placeholders['rewards']] = -np.ones(batch['rewards'].shape)

        resampled_batch_flat = flatten(batch)
        resampled_batch_flat[('observations', 'state_desired_goal')] = (
            resampled_batch_flat[('next_observations', 'state_achieved_goal')])
        resampled_feed_dict = {
            placeholders_flat[key]: resampled_batch_flat[key]
            for key in placeholders_flat.keys()
            if key in resampled_batch_flat.keys()
        }
        resampled_feed_dict[self._placeholders['rewards']] = np.zeros(batch['rewards'].shape)

        feed_dict = {
            key: np.concatenate([
                orig_feed_dict[key], resampled_feed_dict[key]
            ], axis=0)
            for key in orig_feed_dict
        }

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
        diagnostics = super(HERQLearning, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        # samp_obs = batch['observations']
        # batch_size = samp_obs[next(iter(samp_obs))].shape[0]

        # samp_Q_values, samp_Q_losses = self._session.run(
        #     (self._Q_values, self._Q_losses),
        #     feed_dict=super(HERQLearning, self)._get_feed_dict(iteration, batch))
        # goal_Q_values, goal_Q_losses = self._session.run(
        #     (self._Q_values, self._Q_losses),
        #     feed_dict=self._get_goal_feed_dict(batch_size))

        # # TODO: Add validation?
        # diagnostics.update({
        #     'sqil/sample_Q_values': np.mean(samp_Q_values),
        #     'sqil/goal_Q_values': np.mean(goal_Q_values),
        #     'sqil/sample_Q_losses': np.mean(samp_Q_losses),
        #     'sqil/goal_Q_losses': np.mean(goal_Q_losses),
        # })

        return diagnostics
