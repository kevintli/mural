from collections import defaultdict

import numpy as np
from flatten_dict import flatten, unflatten

from softlearning.models.utils import flatten_input_structure
from .pool_sampler import PoolSampler
from scipy import spatial


class NNSampler(PoolSampler):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def initialize_nn_pool(self, nn_pool):
        self._nn_pool = nn_pool

    def initialize(self, env, policy, pool):
        super().initialize(env, policy, pool)

        nn_pool_data = self._nn_pool.batch_by_indices(
            np.arange(self._nn_pool.size))
        nn_pool_obs = nn_pool_data['observations']
        nn_pool_values = [nn_pool_obs[key] for key in self.policy.observation_keys]
        self._flattened_nn_pool_obs = np.concatenate(nn_pool_values, axis=1)
        print("building kd tree")
        self._nn_tree = spatial.KDTree(self._flattened_nn_pool_obs)
        self._cached_nn_inds = (np.zeros(pool.size) - 1).astype('int64')

    def random_batch(self, batch_size=None, **kwargs):
        print('getting nn batch')
        batch_size = batch_size or self._batch_size
        rand_inds = self.pool.random_indices(batch_size)
        rand_batch = self.pool.batch_by_indices(rand_inds, **kwargs)
        rand_obs = rand_batch['observations']
        flattened_obs = np.concatenate(
            [rand_obs[key] for key in self.policy.observation_keys], axis=1
        )

        cached_nn_inds = self._cached_nn_inds[rand_inds]
        batch_to_query_tree = flattened_obs[cached_nn_inds == -1]
        queried_dists, queried_nn_inds = self._nn_tree.query(batch_to_query_tree, eps=1)

        cache_update_inds = rand_inds[cached_nn_inds == -1]
        self._cached_nn_inds[cache_update_inds] = queried_nn_inds
        return self._nn_pool.batch_by_indices(self._cached_nn_inds[rand_inds])
