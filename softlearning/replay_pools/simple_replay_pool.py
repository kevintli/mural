from gym.spaces import Dict
import gzip
import pickle
from .flexible_replay_pool import FlexibleReplayPool, Field


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 obs_save_keys=(),
                 extra_obs_keys_and_fields={},
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space
        self._obs_save_keys = obs_save_keys

        fields = {
            'observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
            },
            'next_observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
            },
            'actions': Field(
                name='actions',
                dtype=action_space.dtype,
                shape=action_space.shape),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }
        fields['observations'].update(extra_obs_keys_and_fields)
        self._obs_save_keys += tuple(extra_obs_keys_and_fields.keys())

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)
        if self._obs_save_keys:
            latest_samples['observations'] = {k: v for k, v in
                                              latest_samples['observations'].items()
                                              if k in self._obs_save_keys}
            latest_samples['next_observations'] = {k: v for k, v in
                                                   latest_samples['next_observations'].items()
                                                   if k in self._obs_save_keys}
        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0
