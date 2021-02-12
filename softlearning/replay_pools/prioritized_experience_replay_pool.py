# from gym.spaces import Dict

from .simple_replay_pool import SimpleReplayPool, Field
from flatten_dict import flatten, unflatten
from .replay_pool import ReplayPool

import numpy as np


class PrioritizedExperienceReplayPool(SimpleReplayPool):
    def __init__(self,
                 mode='moved_screw',
                 per_alpha=1,
                 recompute_priorities_period=10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._priorities = np.zeros(self._max_size)
        self._mode = mode
        self._per_alpha = per_alpha
        self._algorithm = None
        self._recompute_priorities_period = recompute_priorities_period
        self._counter = 0

    def initialize(self, algorithm):
        self._algorithm = algorithm

    def random_indices(self, batch_size):
        if self._size == 0:
            return np.arange(0, 0)
        ## TODO make sampling more efficient if need be via
        # - binary sum tree
        # - cache normalizing constant
        if np.all(self._priorities == 0):
            # initial batch
            indices = np.arange(batch_size)
        else:
            index_distribution = self._priorities[:self._size] / np.sum(
                self._priorities[:self._size])
            indices = np.random.choice(
                self._size, size=batch_size, p=index_distribution)

        if self._mode == "Bellman_Error":
            if self._counter % self._recompute_priorities_period == 0:
                # Recompute priorities of batch sampled uniformly at random from pool
                random_indices = super().random_indices(
                    batch_size * self._recompute_priorities_period)
                bellman_errors = np.squeeze(
                    self._algorithm.get_bellman_error(
                        self.batch_by_indices(random_indices)))
                priorities = bellman_errors ** self._per_alpha
                self._priorities[random_indices] = priorities
            else:
                bellman_errors = np.squeeze(
                    self._algorithm.get_bellman_error(self.batch_by_indices(indices)))
                priorities = bellman_errors ** self._per_alpha
                self._priorities[indices] = priorities
            self._counter += 1
        return indices

    def add_path(self, path):
        ## TODO: check if screw has moved within path
        path = path.copy()

        path_flat = flatten(path)
        path_length = path_flat[next(iter(path_flat.keys()))].shape[0]
        indices = np.arange(
            self._pointer, self._pointer + path_length) % self._max_size

        if self._mode == 'moved_screw':
            initial_corner = path['observations']['in_corner'][0]
            final_corner = path['observations']['in_corner'][-1]

            if initial_corner != final_corner:
                self._priorities[indices] = 100
            elif initial_corner == 0 and final_corner == 0:
                self._priorities[indices] = 10
            else:
                self._priorities[indices] = 1

        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., None],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, None],
        })

        self.add_samples(path)
        if self._mode == 'Bellman_Error':
            path_batch = self.last_n_batch(path_length)
            bellman_errors = np.squeeze(self._algorithm.get_bellman_error(
                path_batch))
            priorities = bellman_errors ** self._per_alpha
            self._priorities[indices] = priorities
