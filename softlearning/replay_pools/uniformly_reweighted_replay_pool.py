from .reweighted_replay_pool import ReweightedReplayPool
import numpy as np
from collections import defaultdict
from flatten_dict import flatten


class UniformlyReweightedReplayPool(ReweightedReplayPool):
    def __init__(self,
                 bin_boundaries,
                 bin_keys, # obs keys to bin on
                 bin_weight_bonus_scaling=0, # exploration bonus
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._bin_boundaries = bin_boundaries
        self._bin_keys = bin_keys # sample keys to bin on
        self._bins = defaultdict(list) # dict: bin_index_tuple->list of sample_inds
        self._reverse_bins = {} # sample_ind -> bin_index_tuple
        self._bin_weight_bonus_scaling = bin_weight_bonus_scaling

    def _set_sample_weights(self, samples, sample_indices):
        flattened_samples = flatten(samples)
        # observation = samples['observations']
        bin_inds = []
        for i, sample_ind in enumerate(sample_indices):
            bin_ind = []
            bin_dim = 0
            for key in self._bin_keys:
                for j in range(flattened_samples[key].shape[1]):
                    bin_ind.append(
                        np.digitize(
                            flattened_samples[key][i, j],
                            self._bin_boundaries[bin_dim]))
                    bin_dim += 1
            bin_ind = tuple(bin_ind)
            bin_inds.append(bin_ind)

            self._bins[bin_ind].append(sample_ind)
            if self._size == self._max_size: # samples are starting to get deleted
                old_bin_ind = self._reverse_bins[sample_ind]
                self._bins[old_bin_ind].remove(sample_ind)
            self._reverse_bins[sample_ind] = bin_ind
        bin_inds = set(bin_inds)

        delta_norm_constant = 0
        for bin_ind in bin_inds:
            sample_inds = self._bins[bin_ind]
            delta_norm_constant -= np.sum(self._unnormalized_weights[sample_inds])
            self._unnormalized_weights[sample_inds] = 1 / len(sample_inds)
            delta_norm_constant += np.sum(self._unnormalized_weights[sample_inds])
        self._normalization_constant += delta_norm_constant

        ## add normalized weights to env_infos
        samples[
            'infos'][
            'reward/normalized_bin_weight_bonus'] = self._unnormalized_weights[
                sample_indices] / self._normalization_constant * self._bin_weight_bonus_scaling

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        """ Modify random training batch by adding an exploration bonus to the rewards. """
        random_indices = self.random_indices(batch_size)
        batch = self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)
        if self._bin_weight_bonus_scaling:
            ## Add bonus to rewards
            normalized_weights = self._unnormalized_weights[random_indices] / self._normalization_constant
            batch['rewards'] += normalized_weights.reshape(-1, 1) * self._bin_weight_bonus_scaling
        return batch
