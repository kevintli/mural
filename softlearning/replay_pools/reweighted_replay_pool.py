import abc
import numpy as np
from .simple_replay_pool import SimpleReplayPool
from flatten_dict import flatten


class ReweightedReplayPool(SimpleReplayPool):
    """Pool which samples transitions proportional to their weights."""
    def __init__(self,
                 mode='moved_screw',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._unnormalized_weights = np.zeros(self._max_size)
        self._normalization_constant = 0
        self._mode = mode
        self._counter = 0

    def random_indices(self, batch_size):
        if self._size == 0:
            return np.arange(0, 0)
        # ## TODO make sampling more efficient if need be via
        # # - binary sum tree
        # # - cache normalizing constant
        if np.all(self._unnormalized_weights == 0):
            # initial batch
            indices = np.arange(batch_size)
        else:
            index_distribution = self._unnormalized_weights[
                :self._size] / self._normalization_constant
            indices = np.random.choice(
                self._size, size=batch_size, p=index_distribution)
        return indices

    def add_path(self, path):
        path = path.copy()

        path_flat = flatten(path)
        path_length = path_flat[next(iter(path_flat.keys()))].shape[0]
        indices = np.arange(
            self._pointer, self._pointer + path_length) % self._max_size

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

        self._set_sample_weights(path, indices)
        self.add_samples(path)

    @abc.abstractmethod
    def _set_sample_weights(self, batch, indices):
        """Compute weights of each sample in the batch and update
        unnormalized weights and normalization constant."""
        pass
