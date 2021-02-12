from collections import defaultdict

import numpy as np
from flatten_dict import flatten, unflatten
import imageio
from softlearning.models.utils import flatten_input_structure
from .base_sampler import BaseSampler
import os

from softlearning.models.state_estimation import normalize

class SimpleSampler(BaseSampler):
    def __init__(self,
                 state_estimator=None,
                 replace_state=False,
                 **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._save_training_video_frequency = 0
        self._images = []

        # self._state_estimator = state_estimator
        # self._replace_state = replace_state

        self._num_high_errors = 0
        self._prefix = np.random.randint(1000)

    @property
    def _policy_input(self):
        try:
            observation = flatten_input_structure({
                key: self._current_observation[key][None, ...]
            for key in self.policy.observation_keys
            })
        except Exception:
            from pprint import pprint; import ipdb; ipdb.set_trace(context=30)

        return observation

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def _get_estimated_state(self, observation):
        assert self._state_estimator is not None, (
            'Need to specify a state estimator in the init'
        )
        assert 'pixels' in self._current_observation, (
            'State estimator only works for pixel observations')
        pixels = self._current_observation['pixels'][None]

        estimated_object_state = self._state_estimator.predict(pixels)[0]
        xy_pos, z_cos, z_sin = (
            normalize(estimated_object_state[:2], -1, 1, -0.1, 0.1),
            estimated_object_state[2][None],
            estimated_object_state[3][None]
        )
        pred_state = {
            'object_xy_position': xy_pos,
            'object_z_orientation_cos': z_cos,
            'object_z_orientation_sin': z_sin
        }
        return pred_state

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
            # Replace the first observation, do all the rest to `next_observation`
            # if self._replace_state:
            #     pred_state = self._get_estimated_state(self._current_observation)
            #     self._current_observation.update(pred_state)

        # if self._state_estimator is not None:
        #     # Save high error images (if there is a label to compare to)
        #     # TODO: Do this for replacing state too
        #     if not self._replace_state:
        #         pred_state = self._get_estimated_state(self._current_observation)
        #         estimated_object_state = np.concatenate(list(pred_state.values()))
        #         self._current_observation.update({
        #             # 'object_xy_position_pred': xy_pos,
        #             # 'object_z_orientation_cos_pred': z_cos,
        #             # 'object_z_orientation_sin_pred': z_sin,
        #             'object_state_prediction': estimated_object_state
        #         })
        #         label = np.concatenate([
        #             # normalize(self._current_observation['object_xy_position'], -0.1, 0.1, -1, 1),
        #             self._current_observation['object_xy_position'],
        #             self._current_observation['object_z_orientation_cos'],
        #             self._current_observation['object_z_orientation_sin'],
        #         ])
        #         if np.linalg.norm(label - estimated_object_state) > 0.1:
        #             self._num_high_errors += 1

        #             imageio.imwrite(
        #                 f'/tmp/test_obs/{self._prefix}_high_error_{self._num_high_errors}.png',
        #                 self._current_observation['pixels'])

        if self._save_training_video_frequency:
            try:
                self._images
            except Exception:
                self._images = []
            self._images.append(
                self.env.render(mode='rgb_array', width=128, height=128))

        action = self.policy.actions_np(self._policy_input)[0]
        next_observation, reward, terminal, info = self.env.step(action)

        # if self._state_estimator is not None and self._replace_state:
        #     pred_next_state = self._get_estimated_state(next_observation)
        #     next_observation.update(pred_next_state)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in flatten(processed_sample).items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = unflatten({
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            })

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
            })

            if self._save_training_video_frequency:
                self._last_n_paths.appendleft({
                    'images': self._images,
                    **last_path,
                })
            else:
                self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self.pool.terminate_episode()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)
            self._images = []

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        # observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(batch_size, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

    def set_save_training_video_frequency(self, flag):
        self._save_training_video_frequency = flag

    def __getstate__(self):
        state = super().__getstate__()
        state['_last_n_paths'] = type(state['_last_n_paths'])((
            type(path)((
                (key, value)
                for key, value in path.items()
                if key != 'images'
            ))
            for path in state['_last_n_paths']
        ))

        del state['_images']
        return state
