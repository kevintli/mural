from gym.spaces import Dict

from .simple_replay_pool import SimpleReplayPool, Field
import numpy as np
from flatten_dict import flatten


class MultiGoalReplayPool(SimpleReplayPool):
    def __init__(self,
                 extra_fields={},
                 *args,
                 **kwargs):
        extra_fields['relabeled'] = Field(
            name='relabeled',
            dtype='bool',
            shape=(1, ))

        super().__init__(extra_fields=extra_fields, *args, **kwargs)

    def add_path(self, path):
        path = path.copy()

        path_flat = flatten(path)
        path_length = path_flat[next(iter(path_flat.keys()))].shape[0]
        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., None],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, None],
            'relabeled': np.array([False]*path_length)[:, None],
        })
        self.add_samples(path)

        path = self._environment.relabel_path(path.copy())
        path_flat = flatten(path)
        path_length = path_flat[next(iter(path_flat.keys()))].shape[0]
        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., None],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, None],
            'relabeled': np.array([True]*path_length)[:, None],
        })

        self.add_samples(path)
