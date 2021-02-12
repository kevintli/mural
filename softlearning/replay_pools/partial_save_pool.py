import gzip
import pickle
from .simple_replay_pool import SimpleReplayPool


class PartialSaveReplayPool(SimpleReplayPool):
    def __init__(self, obs_save_keys, *args, **kwargs):
        self._obs_save_keys = obs_save_keys
        super().__init__(*args, **kwargs)

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)
        latest_samples['observations'] = {k: v for k, v in
                                          latest_samples['observations'].items()
                                          if k in self._obs_save_keys}
        latest_samples['next_observations'] = {k: v for k, v in
                                               latest_samples['next_observations'].items()
                                               if k in self._obs_save_keys}
        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0
