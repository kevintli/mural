from .simple_sampler import SimpleSampler
import numpy as np

class ClassifierSampler(SimpleSampler):
    def __init__(self, algorithm=None, observation_keys=None, **kwargs):
        super().__init__(**kwargs)

        if algorithm:
            assert hasattr(algorithm, 'get_reward'), (
                'Must implement `get_reward` method to save in algorithm'
            )
        self._algorithm = algorithm
        self._observation_keys = observation_keys

    def set_algorithm(self, algorithm):
        if algorithm:
            assert hasattr(algorithm, 'get_reward'), (
                'Must implement `get_reward` method to save in algorithm'
            )

        self._algorithm = algorithm

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        assert self._algorithm, 'Need to set the algorithm first'
        obs_input = {
            key: observation[key][None, ...]
            for key in (self._observation_keys or self.policy.observation_keys)
        }
        learned_reward = self._algorithm.get_reward(obs_input, terminal)
        learned_reward = np.asscalar(learned_reward)
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'learned_rewards': [learned_reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

