from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool, Field


class ActiveReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

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
            'is_goals': Field(
                name='is_goals',
                shape=(1, ),
                dtype='bool'
            ),
            **extra_fields
        }

        super(ActiveReplayPool, self).__init__(
            *args, fields=fields, **kwargs)
