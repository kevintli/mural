import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import tensorflow as tf

from .sac import td_target
from .sac_classifier import SACClassifier
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure

from maml.meta_nml import MetaNML
from model import VAE, MetaMNISTConvModel as MetaConvModel
from notebook_helpers import *
import torch

MAZE_LOW = -6
MAZE_HIGH = 6
N_BINS = (MAZE_HIGH - MAZE_LOW) * 2

class VICE(SACClassifier):
    """Varitational Inverse Control with Events (VICE)

    References
    ----------
    [1] Variational Inverse Control with Events: A General
    Framework for Data-Driven Reward Definition. Justin Fu, Avi Singh,
    Dibya Ghosh, Larry Yang, Sergey Levine, NIPS 2018.
    """

    def __init__(
        self, 
        *args, 
        positive_on_first_occurence=False, 
        active_query_frequency=None,
        gradient_penalty_weight=0,
        death_reward=0,
        sparse_goal=None,
        append_qpos=False,
        use_meta_nml=False,
        use_laplace_smoothing_rewards=False,
        recompute_rewards=False,
        laplace_smoothing_weight=1,
        meta_nml_train_on_positives=False,
        meta_nml_train_every_k=1,
        meta_nml_reset_frequency=0,
        meta_nml_negatives_only=False,
        meta_nml_add_vice_reward=False,
        meta_nml_do_metatraining=True,
        meta_nml_shuffle_states=False,
        n_initial_classifier_train_steps=1,
        meta_nml_checkpoint_frequency=100,
        meta_nml_layers=[2048, 2048],
        meta_nml_num_finetuning_layers=None,
        meta_nml_uniform_train_data=False,
        meta_nml_use_preprocessor=False,
        meta_nml_custom_embedding_key=None,
        meta_nml_use_vae=False,
        meta_nml_vae_model=None,
        meta_nml_train_vae=False,
        meta_nml_train_embedding_frequency=None,
        num_initial_train_embedding_epochs=800,
        meta_nml_train_embedding_epochs=50,
        meta_nml_reward_type='probs',  # Options are {'probs', 'log_probs'}
        meta_train_sample_size=128, 
        meta_test_sample_size=1000, 
        meta_task_batch_size=1, 
        accumulation_steps=16,
        meta_test_batch_size=16, 
        num_initial_meta_epochs=10,
        num_meta_epochs=1, 
        nml_grad_steps=1,
        points_per_meta_task=1,
        equal_pos_neg_test=True, 
        dist_weight_thresh=None, 
        query_point_weight=1, 
        test_strategy='all', 
        **kwargs
    ):
        self._active_query_frequency = active_query_frequency
        self._gradient_penalty_weight = gradient_penalty_weight
        self._append_qpos = append_qpos
        self._sparse_goal = sparse_goal
        self._death_reward = death_reward
        self._use_laplace_smoothing_rewards = use_laplace_smoothing_rewards
        self._recompute_rewards = recompute_rewards
        self._laplace_smoothing_weight = laplace_smoothing_weight
        self._use_meta_nml = use_meta_nml
        self._meta_nml_checkpoint_frequency = meta_nml_checkpoint_frequency
        self._meta_nml_train_every_k = meta_nml_train_every_k
        self._meta_nml_reward_type = meta_nml_reward_type
        self._meta_nml_reset_frequency = meta_nml_reset_frequency
        self._meta_nml_train_on_positives = meta_nml_train_on_positives
        self._meta_nml_uniform_train_data = meta_nml_uniform_train_data
        self._meta_nml_negatives_only = meta_nml_negatives_only
        self._meta_nml_add_vice_reward = meta_nml_add_vice_reward
        self._meta_nml_shuffle_states = meta_nml_shuffle_states
        self._meta_nml_use_preprocessor = meta_nml_use_preprocessor
        self._meta_nml_custom_embedding_key = meta_nml_custom_embedding_key
        self._meta_nml_use_vae = meta_nml_use_vae
        self._meta_nml_train_vae = meta_nml_train_vae
        self._meta_nml_train_embedding_frequency = meta_nml_train_embedding_frequency
        self._meta_nml_train_embedding_epochs = meta_nml_train_embedding_epochs
        self._num_initial_train_embedding_epochs = num_initial_train_embedding_epochs
        self._n_initial_classifier_train_steps = n_initial_classifier_train_steps

        super(VICE, self).__init__(*args, **kwargs)

        print(f"[VICE] Epoch length: {self._epoch_length}")
        
        if self._use_laplace_smoothing_rewards:
            self.rewards_matrix = np.repeat(0.5, N_BINS * N_BINS).reshape((N_BINS, N_BINS))

        self._from_vision = 'pixels' in self._classifier.observation_keys

        if self._use_meta_nml:
            vae_kwargs = {}
            input_dim = 0
            if self._from_vision:
                # Using a convolutional network for image inputs
                vae_kwargs['model'] = MetaConvModel(2, in_channels=3)
                print(f"[MetaNML] Using convolutional network for classifier")
                print(vae_kwargs['model'])
            else:
                # Using a feedforward network. Figure out what the input dimension
                # should be based on the classifier observation keys (or preprocessor embedding dim)
                if self._meta_nml_use_preprocessor:
                    assert hasattr(self._classifier, "observations_preprocessors"), \
                        "Must define a preprocessor for the reward classifier" \
                        + " when setting meta_nml_use_preprocessor=True"
                    self._preprocessor = tf.keras.Model(inputs=self._classifier.layers[0].input, outputs=self._classifier.layers[1].output)
                    self._init_preprocessor_outputs()
                    input_dim = self._classifier.layers[1].output.shape[-1]
                else:
                    input_dim = self._concat_classifier_obs({
                        key: self._goal_examples[key][0,None]
                        for key in self._goal_examples
                    }).shape[-1]
                print(f"[MetaNML] Using feedforward classifier, input dimension {input_dim}")

            if meta_nml_vae_model:
                model_vae = VAE(img_channels=3)
                model_vae.load_state_dict(torch.load(meta_nml_vae_model))
                model_vae.cuda()
                vae_kwargs['model_vae'] = model_vae
                vae_kwargs['embedding_type'] = 'vae'
                vae_kwargs['num_finetuning_layers'] = 2
                print(f"[MetaNML] Loaded VAE weights from {meta_nml_vae_model}")

            if meta_nml_custom_embedding_key:
                print(f"[MetaNML] Using custom embedding key: {meta_nml_custom_embedding_key}")
                vae_kwargs['embedding_type'] = 'custom'

            self.meta_nml_kwargs = {
                **vae_kwargs,
                "hidden_sizes": list(meta_nml_layers),
                "input_dim": input_dim, 
                "points_per_task": points_per_meta_task,
                "equal_pos_neg_test": equal_pos_neg_test and not meta_nml_negatives_only, 
                "dist_weight_thresh": dist_weight_thresh,
                "query_point_weight": query_point_weight,
                "do_metalearning": meta_nml_do_metatraining,
                "train_vae": meta_nml_train_vae,
                "num_finetuning_layers": meta_nml_num_finetuning_layers,
            }

            self.meta_nml = MetaNML(**self.meta_nml_kwargs)
            self._meta_train_sample_size = meta_train_sample_size
            self._meta_test_sample_size = meta_test_sample_size
            self._meta_task_batch_size = meta_task_batch_size
            self._accumulation_steps = accumulation_steps
            self._meta_test_batch_size = meta_test_batch_size
            self._num_initial_meta_epochs = num_initial_meta_epochs
            self._num_meta_epochs = num_meta_epochs
            self._nml_grad_steps = nml_grad_steps
            self._test_strategy = test_strategy

        self._positive_on_first_occurence = positive_on_first_occurence
        if positive_on_first_occurence:
            env = self._training_environment.unwrapped
            # self._seen_states = set()
            self._seen_states = [
                [False for _ in range(env.n_bins + 1)]
                for _ in range(env.n_bins + 1)
            ]  

#    def _init_extrinsic_reward(self):
#        classifier_inputs = flatten_input_structure({
#            name: tf.exp(self._placeholders['observations'][name])
#            for name in self._classifier.observation_keys
#        })
#        observation_log_p = self._classifier(classifier_inputs)
#        self._reward_t = self._unscaled_ext_reward = observation_log_p

    def _use_fixed_rewards(self):
        return self._use_meta_nml or self._use_laplace_smoothing_rewards

    def _init_classifier_reward(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier.observation_keys
        })
        observation_logits = self._classifier(classifier_inputs)

        if self._meta_nml_reward_type == 'logits':
            self._clf_reward = observation_logits
        elif self._meta_nml_reward_type == 'probs':
            self._clf_reward = tf.nn.sigmoid(observation_logits)
        else:
            raise NotImplementedError(
                f"Unknown meta-NML reward type: {self._meta_nml_reward_type}")

    def _init_preprocessor_outputs(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier.observation_keys
        })
        self._preprocessor_outputs = self._preprocessor(classifier_inputs)

    def _init_extrinsic_reward(self):
        if self._use_fixed_rewards():
            if self._meta_nml_add_vice_reward or self._meta_nml_use_preprocessor:
                self._init_classifier_reward()

            # Don't include classifier reward evaluation in the computation graph,
            # because meta-NML is very expensive. Instead, we'll compute the rewards 
            # directly when the states are sampled (by using ClassifierSampler),
            # and store them as `learned_rewards` in the replay pool.
            self._reward_t = self._unscaled_ext_reward = self._placeholders['learned_rewards']
        else:
            super(VICE, self)._init_extrinsic_reward()

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(VICE, self)._get_feed_dict(iteration, batch)
        if 'learned_rewards' in batch:
            if self._recompute_rewards:
                # Recompute rewards with the most recent classifier params
                feed_dict[self._placeholders['learned_rewards']] = self.get_reward(batch['observations'])[:,None]
        # else:
        #         # Include the stale classifier rewards, which were assigned when states were first encountered
        #         feed_dict[self._placeholders['learned_rewards']] = batch['learned_rewards']
        return feed_dict

    def _get_episode_reward_feed_dict(self, episodes):
        feed_dict = super(VICE, self)._get_episode_reward_feed_dict(episodes)
        if self._use_fixed_rewards():
            feed_dict[self._placeholders['learned_rewards']] = np.concatenate([
                episode['learned_rewards'] for episode in episodes
            ])
        return feed_dict

    def _get_classifier_feed_dict(self):
        negatives = self.sampler.random_batch(
            self._classifier_batch_size
        )['observations']

        if self._positive_on_first_occurence:
            # Still things left to explore
            env = self._training_environment.unwrapped
            first_occ_idxs = []
            for i in range(len(negatives[next(iter(negatives))])):
                x_d, y_d = env._discretize_observation({
                    key: val[i]
                    for key, val in negatives.items()
                })
                if not self._seen_states[x_d][y_d]:
                # if (x_d, y_d) not in self._seen_states:
                    first_occ_idxs.append(i)
                    # self._seen_states.add((x_d, y_d))
                    self._seen_states[x_d][y_d] = True

        # DEBUG: Testing with the same negatives pool for each training iteration
        # negatives = type(self._pool.data)(
        #     (key[1], value[:self._classifier_batch_size])
        #     for key, value in self._pool.data.items()
        #     if key[0] == 'observations')

        rand_positive_ind = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=self._classifier_batch_size)
        positives = {
            key: values[rand_positive_ind]
            for key, values in self._goal_examples.items()
        }
        if self._positive_on_first_occurence:
            positives = {
                key: np.concatenate([val, negatives[key][first_occ_idxs]], axis=0)
                for key, val in positives.items()
            }
            labels_batch = np.zeros(
                (self._classifier_batch_size +
                (self._classifier_batch_size + len(first_occ_idxs)), 2),
                dtype=np.int32)
            labels_batch[:self._classifier_batch_size, 0] = 1
            labels_batch[self._classifier_batch_size:, 1] = 1

        else:
            labels_batch = np.zeros(
                (2 * self._classifier_batch_size, 2),
                dtype=np.int32)
            labels_batch[:self._classifier_batch_size, 0] = 1
            labels_batch[self._classifier_batch_size:, 1] = 1

        observations_batch = {
            key: np.concatenate((negatives[key], positives[key]), axis=0)
            # for key in self._classifier.observation_keys
            for key in self._policy.observation_keys
        }

        if self._mixup_alpha > 0:
            observations_batch, labels_batch = mixup(
                observations_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observations_batch[key]
                # for key in self._classifier.observation_keys
                for key in self._policy.observation_keys
            },
            self._placeholders['labels']: labels_batch,
        }

        return feed_dict

    def _init_placeholders(self):
        super()._init_placeholders()
        self._placeholders['labels'] = tf.placeholder(
            tf.int32,
            shape=(None, 2),
            name='labels',
        )
        if self._use_fixed_rewards():
            self._placeholders['learned_rewards'] = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='learned_rewards',
            )

    def _init_classifier_update(self):
        if (self._use_fixed_rewards() and not self._meta_nml_add_vice_reward 
            and not self._meta_nml_use_preprocessor):
            return

        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier.observation_keys
        })
        log_p = self._classifier(classifier_inputs)
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        sampled_actions = self._policy.actions(policy_inputs)
        log_pi = self._policy.log_pis(policy_inputs, sampled_actions)
        # pi / (pi + f), f / (f + pi)
        log_pi_log_p_concat = tf.concat([log_pi, log_p], axis=1)

        self._classifier_loss_t = tf.reduce_mean(
            tf.compat.v1.losses.softmax_cross_entropy(
                self._placeholders['labels'],
                log_pi_log_p_concat,
            )
        )
        self._classifier_training_op = self._get_classifier_training_op()

    def _concat_classifier_obs(self, batch, shuffle=True):
        if self._meta_nml_shuffle_states and shuffle:
            states = discretized_states(
                batch['state_observation'], 
                bins=N_BINS, low=MAZE_LOW, high=MAZE_HIGH
            )
            results = (np.hstack([self._random_mapping_x[states[:,0]][:,None], self._random_mapping_y[states[:,1]][:,None]]) - 8) / 2
            return results
        
        if self._meta_nml_use_vae or self._meta_nml_custom_embedding_key:
            return batch['pixels'].transpose((0, 3, 1, 2))

        return np.hstack([batch[name] for name in self._classifier.observation_keys])

    def _get_laplace_obs(self, batch):
        if self._meta_nml_use_vae or self._meta_nml_custom_embedding_key:
            return batch['state_observation']
        else:
            return self._concat_classifier_obs(batch, shuffle=False)

    def _sample_negatives(self, size):
        batch = self.sampler.random_batch(size)['observations']
        if self._meta_nml_use_preprocessor:
            # If using a preprocessor, we train & evaluate meta-NML on the embedding 
            # given by the preprocessor instead of the raw inputs.
            negatives = self._compute_preprocessor_embedding(batch)
        else:
            negatives = self._concat_classifier_obs(batch)
        labels = np.zeros(len(negatives))

        if self._meta_nml_custom_embedding_key:
            return negatives.astype(np.float32), labels, batch[self._meta_nml_custom_embedding_key]
        else:
            return negatives.astype(np.float32), labels

    def _sample_positives(self, size, shuffle=True, for_laplace=False):
        rand_positive_ind = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=size)

        if for_laplace:
            return self._goal_examples['state_observation'][rand_positive_ind]

        batch = {
            key: values[rand_positive_ind]
            for key, values in self._goal_examples.items() 
                if key in self._classifier.observation_keys
        }

        if self._meta_nml_use_preprocessor:
            # If using a preprocessor, we train & evaluate meta-NML on the embedding 
            # given by the preprocessor instead of the raw inputs.
            positives = self._compute_preprocessor_embedding(batch)
        else:
            positives = self._concat_classifier_obs(batch, shuffle=shuffle)

        if self._meta_nml_custom_embedding_key:
            return (positives.astype(np.float32), np.ones(len(positives)), 
                self._goal_examples[self._meta_nml_custom_embedding_key][rand_positive_ind])
        else:
            return positives.astype(np.float32), np.ones(len(positives))

    def _sample_meta_test_batch(self, size):
        if self._meta_nml_negatives_only:
            return self._sample_negatives(size)
        else:
            negatives = self._sample_negatives(size // 2)
            positives = self._sample_positives(size // 2)
            return tuple(np.concatenate([a, b], axis=0) for a, b in zip(negatives, positives))

    def _get_grid_vals(self):
        n_bins = (MAZE_HIGH - MAZE_LOW) * 2
        grid_vals = np.arange(n_bins * n_bins)[:,None]
        grid_vals = np.hstack((grid_vals % n_bins, grid_vals // n_bins)) / 2 - (MAZE_HIGH - MAZE_LOW) // 2
        return grid_vals

    def _get_laplace_rewards(self, states=None):
        # if states is None:
        #     states = self._get_grid_vals()
        # env = self._training_environment._env.unwrapped
        # discrete_x, discrete_y = env._discretize_observation({
        #     'state_observation': [states[:,0], states[:,1]]
        # })
        # discrete_obs = np.hstack((discrete_x[:,None], discrete_y[:,None]))
        # laplace_rewards = env.compute_rewards(None, 
        #     {
        #         'state_achieved_goal': states,
        #         'discrete_observation': discrete_obs,
        #         'state_desired_goal': np.array([3., 3.]),
        #     }, 
        #     reward_type='laplace_smoothing'
        # )
        """
        Compute a N_BINS x N_BINS rewards matrix using all the current visitations in the replay pool
        and an equal number of positives
        """
        if self._meta_nml_use_vae or self._meta_nml_custom_embedding_key:
            negatives = self.sampler.pool.last_n_batch(5e5)['observations']['state_observation']
            positives = self._sample_positives(len(negatives), for_laplace=True, shuffle=False)
        else:
            negatives = self._concat_classifier_obs(self.sampler.pool.last_n_batch(5e5)['observations'], shuffle=False)
            positives, _ = self._sample_positives(len(negatives), shuffle=False)

        if self._append_qpos:
            # Only use xy state for computing visitations
            negatives = negatives[:,:2]
            positives = positives[:,:2]

        states = np.vstack((negatives, positives))
        labels = np.hstack((np.zeros(len(negatives)), np.ones(len(positives))))
        laplace_rewards = true_bayesian_vice_reward(states, labels, low=MAZE_LOW, high=MAZE_HIGH, bins=N_BINS, 
            pos_weight=self._laplace_smoothing_weight, neg_weight=self._laplace_smoothing_weight)
        grid_vals = self._get_grid_vals()
        laplace_rewards += self._get_sparse_rewards(grid_vals).reshape(N_BINS, N_BINS)
        if self._meta_nml_reward_type == 'log_probs':
            return np.log(laplace_rewards)
        elif self._meta_nml_reward_type == 'sqrt_probs':
            return np.sqrt(laplace_rewards)
        else:
            return laplace_rewards

    def _get_image_obs(self, states):
        env = self._training_environment.unwrapped
        result = []
        old_position = env._position
        for state in states:
            env._position = state
            result.append(env.render(mode='rgb_array', width=28, height=28, invert_colors=True))
        env._position = old_position
        return np.array(result)

    def _get_full_state(self, xy):
        NOISE = 1
        DEFAULT_QPOS = np.array([5.5, 4.5, 0.565, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.]).astype(np.float32)
    
        if len(xy.shape) == 1:
            return np.hstack((xy, DEFAULT_QPOS[2], DEFAULT_QPOS[3:] + np.random.uniform(0, NOISE, size=(12,)))).astype(np.float32)
        return np.hstack((xy, np.repeat(DEFAULT_QPOS[2][None][None], len(xy), axis=0), 
                                np.repeat(DEFAULT_QPOS[3:][None], len(xy), axis=0) 
                                    + np.random.uniform(0, NOISE, size=(len(xy), 12)))).astype(np.float32)

    def _plot_ant_dataset(self, dataset):
        X, y = dataset
        positives = X[y == 1]
        negatives = X[y == 0]
        plt.scatter(negatives[:,0], negatives[:,1], marker='x', color='r', s=50)
        plt.scatter(positives[:,0], positives[:,1], marker='*', color='g', s=50)
        plt.gca().add_patch(patches.Rectangle(((-0.75, -2)),1.5,8,linewidth=1,edgecolor='k',facecolor='k'))
        plt.gca().set_aspect("equal")
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

    def _plot_maze_rewards(self, train_data=None, flip_y=False):
        grid_vals = self._get_grid_vals()
        grid_vals_state = grid_vals
        n_bins = int(len(grid_vals) ** 0.5)

        if self._meta_nml_use_preprocessor or self._meta_nml_use_vae or self._meta_nml_custom_embedding_key:
            grid_vals = {'pixels': self._get_image_obs(grid_vals), 'state_observation': grid_vals}
        elif self._append_qpos:
            grid_vals = {self._classifier.observation_keys[0]: self._get_full_state(grid_vals)}
        else:
            grid_vals = {self._classifier.observation_keys[0]: grid_vals}

        rewards = self.get_reward(grid_vals).reshape(n_bins, n_bins)
        laplace_rewards = self._get_laplace_rewards().reshape(n_bins, n_bins) 

        flip_y = flip_y or self._append_qpos

        if self._meta_nml_add_vice_reward or self._meta_nml_train_vae:
            fig, ((ax0, ax1, ax2, ax3), (_, ax4, ax5, ax6)) = plt.subplots(2, 4, figsize=(20, 10))
        else:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 5))

        last_epoch_batch = self.sampler.pool.last_n_batch(self._epoch_length)
        key = 'state_observation' if (self._meta_nml_use_vae or self._meta_nml_custom_embedding_key) \
             else self._classifier.observation_keys[0]
        epoch_states = last_epoch_batch['observations'][key]
        epoch_rewards = last_epoch_batch['learned_rewards']

        if self._append_qpos:
            # Only use xy state for plotting
            epoch_states = epoch_states[:,:2]
            train_data_xy = (train_data[0][:,:2], train_data[1])
        else:
            train_data_xy = train_data

        if flip_y:
            rewards = rewards[::-1,:]
            laplace_rewards = laplace_rewards[::-1,:]

        im = ax0.scatter(epoch_states[:,0], epoch_states[:,1], c=epoch_rewards.squeeze())
        if self._meta_nml_reward_type == 'log_probs':
            im.set_clim(-10, 0)
        else:
            im.set_clim(0, 1)
        plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        # Plot start and end of each trajectory
        max_l = self.sampler._max_path_length
        ax0.scatter(epoch_states[0::max_l,0], epoch_states[0::max_l,1], marker='*', s=100, color='m')
        ax0.scatter(epoch_states[max_l-1::max_l,0], epoch_states[max_l-1::max_l,1], marker='x', s=100, color='r')
        ax0.set_ylim(MAZE_LOW, MAZE_HIGH)
        ax0.set_xlim(MAZE_LOW, MAZE_HIGH)
        ax0.set_aspect("equal")
        if not flip_y:
            ax0.invert_yaxis()
        ax0.set_title(f"Epoch {self._epoch} visitations")

        plt.sca(ax1)
        self._plot_ant_dataset(train_data_xy)
        plt.title("Sampled meta-tasks")

        im = ax2.imshow(rewards)
        if self._meta_nml_reward_type == 'log_probs':
            im.set_clim(-10, 0)
        else:
            im.set_clim(0, 1)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("Meta-NML rewards")

        im = ax3.imshow(laplace_rewards)
        if self._meta_nml_reward_type == 'log_probs':
            im.set_clim(-10, 0)
        else:
            im.set_clim(0, 1)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("Laplace smoothing rewards")

        # im = ax3.imshow(np.clip(rewards, 0, None) - laplace_rewards, cmap='RdBu')
        # if self._meta_nml_reward_type == 'log_probs':
        #     im.set_clim(-5, 5)
        # else:
        #     im.set_clim(-0.5, 0.5)
        # plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        # ax3.set_title("Errors (meta-NML - Laplace smoothing)")

        if self._meta_nml_add_vice_reward:
            clf_rewards = self._session.run(
                self._clf_reward,
                feed_dict={self._placeholders['observations']['state_observation']: grid_vals}
            )[:,0].reshape(n_bins, n_bins)

            im = ax4.imshow(rewards - clf_rewards)
            if self._meta_nml_reward_type == 'log_probs':
                im.set_clim(-10, 0)
            else:
                im.set_clim(0, 1)
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            ax4.set_title("Meta-NML rewards")

            im = ax5.imshow(clf_rewards)
            if self._meta_nml_reward_type == 'log_probs':
                im.set_clim(-10, 0)
            else:
                im.set_clim(0, 1)
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            ax5.set_title("VICE rewards")

            im = ax6.imshow(rewards)
            if self._meta_nml_reward_type == 'log_probs':
                im.set_clim(-10, 0)
            else:
                im.set_clim(0, 1)
            plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            ax6.set_title("Combined rewards")

        if self._meta_nml_train_vae:
            self.meta_nml.model_vae.cuda()
            grid_features = self.meta_nml.model_vae(torch.Tensor(grid_vals['pixels']).cuda())[1].cpu().detach().numpy()
            goal_features = self.meta_nml.model_vae(torch.Tensor(self._get_image_obs(np.array([3, 3])[None])).cuda())[1].cpu().detach().numpy()
            distances = np.linalg.norm(grid_features - goal_features, axis=-1).reshape(n_bins, n_bins)
            ax4.set_aspect("equal")
            # ax4.invert_yaxis()
            im = ax4.contourf(grid_vals_state[:,0].reshape(n_bins, n_bins), grid_vals_state[:,1].reshape(n_bins, n_bins), distances, cmap='viridis_r', levels=30)
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            ax4.set_title("VAE Embedding Distances")

            if hasattr(self, "_last_vae_losses"):
                losses, nlls, klds = self._last_vae_losses, self._last_vae_nlls, self._last_vae_klds
                ax5.plot(losses[10:], color='tab:blue', label='Total Loss')
                ax5.plot(nlls[10:], color='tab:orange', label='NLL')
                ax5.plot(klds[10:], color='tab:green', label='KLD')
                ax5.legend()
                ax5.set_ylim(0, 50)
                ax5.set_title(f"VAE training losses from epoch {self._epoch // 10 * 10}")

        plt.show()
        save_dir = './comparisons/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, 'comparison_%d.png' % self._epoch))

    def _convert_to_states(self, images):
        states = []

        for x in images:
            idxs = np.where(np.logical_and(x[2] == 1, x[0] == 0))
        #     print(idxs)
            i = min(4, len(idxs[0]) - 1)
            pixel_y, pixel_x = idxs[0][i], idxs[1][i]
            state_x, state_y = pixel_x / 28 * 8 - 4, pixel_y / 28 * 8 - 4
            states.append([state_x, state_y])

        return np.array(states)

    def _convert_to_images(self, states, env):
        if isinstance(states, list):
            states = np.array(states)
            
        result = []
        old_position = env._position
        for state in states:
            env._position = state
            result.append(env.render(mode='rgb_array', width=28, height=28, invert_colors=True))
        env._position = old_position
        return np.array(result).astype(np.float32).transpose((0, 3, 1, 2))

    def _epoch_after_hook(self, *args, **kwargs):
        if self._active_query_frequency and self._epoch % self._active_query_frequency == 0:
            self._do_active_query()


        #########################################
        # Sample training data (before plotting)
        #########################################
        if self._use_meta_nml and self._epoch % self._meta_nml_train_every_k == 0:
            if self._meta_nml_train_on_positives:
                train_data = self._sample_meta_test_batch(self._meta_train_sample_size)
            else:
                train_data = self._sample_negatives(self._meta_train_sample_size)
            test_data = self._sample_meta_test_batch(self._meta_test_sample_size)

        ###################################
        # Plot rewards and other metrics
        # FOR 2D MAZE ONLY!!
        ###################################
        # from IPython import embed; embed()
        # if self._use_fixed_rewards():
        #     self._plot_maze_rewards(train_data=train_data if not self._meta_nml_use_vae else (self._convert_to_states(train_data[0]), train_data[1]))

        if self._use_meta_nml and self._epoch % self._meta_nml_train_every_k == 0:
            ###################################
            # Save model checkpoint
            ###################################
            if self._epoch % self._meta_nml_checkpoint_frequency == 0:
                ckpt_dir = './meta_nml_models'
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                
                ckpt_path = os.path.join(ckpt_dir, f'checkpoint-{self._epoch}.pt')
                with open(ckpt_path, 'wb') as f:
                    torch.save(self.meta_nml.model.state_dict(), f)
                print(f"Saved meta-NML model to {ckpt_path}")

            # if self._meta_nml_uniform_train_data:
            #     negatives = self.sampler.pool.last_n_batch(15e5)['observations']['state_observation']
            #     positives = self._sample_positives(len(negatives), for_laplace=True, shuffle=False)
            #     sampled_states = np.vstack((negatives, positives))
            #     visitations = np.clip(discrete_to_counts(discretized_states(sampled_states)).T, 0, 1)
            #     grid_vals = get_grid_vals()
            #     xs, ys = grid_vals[:,0].reshape(16, 16), grid_vals[:,1].reshape(16, 16)
            #     support = np.hstack((xs[visitations > 0][:,None], ys[visitations > 0][:,None])).astype(np.float32)

            #     plt.figure()
            #     plot_dataset((support, np.zeros(len(support))))
            #     save_dir = './train_data/'
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     plt.savefig(os.path.join(save_dir, 'train_%d.png' % self._epoch))

            #     train_data = (self._convert_to_images(support, self._training_environment.unwrapped), np.zeros(len(support)))
            # else:
            #     plt.figure()
            #     plot_dataset((self._convert_to_states(train_data[0]), train_data[1]))
            #     save_dir = './train_data/'
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     plt.savefig(os.path.join(save_dir, 'train_%d.png' % self._epoch))

            # plt.figure()
            # plt.imshow(train_data[0].sum(axis=0).transpose((1, 2, 0)))
            # save_dir = './train_data/'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # plt.savefig(os.path.join(save_dir, 'train_%d.png' % self._epoch))

            data_dir = './meta_nml_data/'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            if self._meta_nml_custom_embedding_key:
                np.savez(os.path.join(data_dir, f'data_{self._epoch}.npz'), 
                    train_X=train_data[0], train_y=train_data[1], train_embedding=train_data[2],
                    test_X=test_data[0], test_y=test_data[1], test_embedding=test_data[2])
            else:
                np.savez(os.path.join(data_dir, f'data_{self._epoch}.npz'), 
                    train_X=train_data[0], train_y=train_data[1],
                    test_X=test_data[0], test_y=test_data[1])

            ###################################
            # Train meta-NML embedding
            ###################################
            if (self._meta_nml_train_embedding_frequency and 
                self._epoch % self._meta_nml_train_embedding_frequency == 0):
                self._last_vae_losses, self._last_vae_nlls, self._last_vae_klds = self.meta_nml.train_embedding(
                    test_data, num_epochs=self._num_initial_train_embedding_epochs if self._epoch == 0 \
                        else self._meta_nml_train_embedding_epochs)

            ###################################
            # Train meta-NML classifier
            ###################################
            # Meta-NML probs may take several iterations at first to become accurate,
            # but afterwards they tend not to require much training each time
            num_epochs = self._num_initial_meta_epochs if self._epoch == 0 \
                else self._num_meta_epochs

            # Reset periodically to avoid getting stuck with poor weights
            if self._meta_nml_reset_frequency and (self._epoch + 1) % self._meta_nml_reset_frequency == 0:
                self.meta_nml = MetaNML(**self.meta_nml_kwargs)
                num_epochs = self._num_initial_meta_epochs

            self.meta_nml.train(train_data, test_data, 
                batch_size=self._meta_task_batch_size, accumulation_steps=self._accumulation_steps,
                num_epochs=num_epochs, test_strategy=self._test_strategy, 
                test_batch_size=self._meta_test_batch_size, mixup_alpha=self._mixup_alpha)

        if self._use_laplace_smoothing_rewards:
            # Compute a fixed rewards matrix to use for the next epoch
            self.rewards_matrix = self._get_laplace_rewards()

        if not self._use_fixed_rewards() or self._meta_nml_add_vice_reward or self._meta_nml_use_preprocessor:
            num_train_steps = self._n_initial_classifier_train_steps if self._epoch == 0 \
                else self._n_classifier_train_steps
            for i in range(num_train_steps):
                feed_dict = self._get_classifier_feed_dict()
                self._train_classifier_step(feed_dict)

    def _do_active_query(self):
        batch_of_interest = self._pool.last_n_batch(
            self._epoch_length * self._active_query_frequency)
        observations_of_interest = batch_of_interest['observations']
        labels_of_interest = batch_of_interest['rewards'] == 1
        rewards_of_interest = self.get_reward(observations_of_interest)

        # TODO: maybe log this quantity
        max_ind = np.argmax(rewards_of_interest)
        if labels_of_interest[max_ind]:
            self._goal_examples = {
                key: np.concatenate((
                    self._goal_examples[key],
                    observations_of_interest[key][[max_ind]],
                ))
                for key in self._goal_examples.keys()
            }

    def _compute_preprocessor_embedding(self, observations):
        return self._session.run(
            self._preprocessor_outputs,
            feed_dict={
                self._placeholders['observations'][name]: observations[name]
                for name in self._classifier.observation_keys
            }
        )

    def get_reward(self, observations, terminals=None):
        """
        Computes the reward for a batch of observations encountered by ClassifierSampler.

        Currently used only for the meta-NML version, since we don't want to recompute
        rewards every time a training batch is sampled from the replay buffer
        """
        if self._use_laplace_smoothing_rewards:
            states = discretized_states(
                self._concat_classifier_obs(observations), 
                bins=N_BINS, low=MAZE_LOW, high=MAZE_HIGH
            )
            learned_reward = self.rewards_matrix[states[:,1], states[:,0]]
        elif self._use_meta_nml:
            if self._epoch == 0:
                finetuning_sample = None
            else:
                finetuning_sample = self._sample_meta_test_batch(self._meta_test_sample_size)

            if self._meta_nml_use_preprocessor:
                # If using a preprocessor, we train & evaluate meta-NML on the embedding 
                # given by the preprocessor instead of the raw inputs.
                classifier_inputs = self._compute_preprocessor_embedding(observations)
            else:
                classifier_inputs = self._concat_classifier_obs(observations)

            # import ipdb; ipdb.set_trace()
            if self._meta_nml_custom_embedding_key:
                eval_inputs = (classifier_inputs, observations[self._meta_nml_custom_embedding_key])
            else:
                eval_inputs = classifier_inputs
            learned_reward = self.meta_nml.evaluate(eval_inputs,
                num_grad_steps=self._nml_grad_steps, train_data=finetuning_sample)[:,1]

            if self._meta_nml_add_vice_reward:
                # Add the rewards from regular VICE (for shaping near the goal).
                # NOTE: this assumes the VICE classifier is trained for enough steps that it's 
                #  mostly 0 everywhere else, so it doesn't affect the meta-NML rewards.
                learned_reward += self._session.run(
                    self._clf_reward,
                    feed_dict={
                        self._placeholders['observations'][name]: observations[name]
                        for name in self._classifier.observation_keys
                    }
                )[:,0]

            if self._meta_nml_reward_type == 'probs':
                pass
            elif self._meta_nml_reward_type == 'log_probs':
                learned_reward = np.log(learned_reward)
            elif self._meta_nml_reward_type == 'sqrt_probs':
                learned_reward = np.sqrt(learned_reward)
            else:
                raise Exception(f"Unrecognized meta-NML reward type: {self._meta_nml_reward_type}")
        else:
            learned_reward = self._session.run(
                self._reward_t,
                feed_dict={
                    self._placeholders['observations'][name]: observations[name]
                    for name in self._policy.observation_keys
                    # for name in self._classifiers[0].observation_keys
                }
            )

        if self._death_reward and terminals is not None:
            learned_reward[terminals == True] += self._death_reward

        # learned_reward += self._get_sparse_rewards(classifier_inputs)

        return learned_reward

    def _get_sparse_rewards(self, states):
        reward = np.zeros(len(states))
        if self._sparse_goal:
            # Set reward for anything within threshold of goal(s) to 1
            # (ignore any classifier reward)
            dists = []
            goals, threshold = self._sparse_goal
            if len(goals.shape) == 1:
                goals = goals[None]
            for goal in goals:
                dist = np.linalg.norm(states - goal, axis=-1)
                dists.append(dist)
            min_dist = np.array(dists).min(axis=0)
            reward[min_dist <= threshold] = 1
        return reward

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        if self._use_fixed_rewards():
            # TODO: make diagnostics work with meta-NML version.
            # Not compatible right now because we don't use the same classifier outputs
            # and reward computation is not done in Tensorflow
            return diagnostics

        sample_observations = batch['observations']
        sample_actions = batch['actions']
        num_sample_observations = sample_observations[
            next(iter(sample_observations))].shape[0]
        sample_labels = np.repeat(((1, 0), ), num_sample_observations, axis=0)

        goal_index = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations = {
            key: values[goal_index] for key, values in self._goal_examples.items()
            if key in self._policy.observation_keys
        }
        # Sample goal actions uniformly in action space
        # action_space_dim = sample_actions.shape[1]
        # goal_actions = np.random.uniform(
        #     low=-1, high=1, size=(num_sample_observations, action_space_dim))
        # goal_validation_actions = np.random.uniform(
        #     low=-1, high=1, size=(num_sample_observations, action_space_dim))
        goal_index_validation = np.random.randint(
            self._goal_examples_validation[
                next(iter(self._goal_examples_validation))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations_validation = {
            key: values[goal_index_validation]
            for key, values in self._goal_examples_validation.items()
            if key in self._policy.observation_keys
        }

        num_goal_observations = goal_observations[
            next(iter(goal_observations))].shape[0]
        goal_labels = np.repeat(((0, 1), ), num_goal_observations, axis=0)

        num_goal_observations_validation = goal_observations_validation[
            next(iter(goal_observations_validation))].shape[0]
        goal_validation_labels = np.repeat(
            ((0, 1), ), num_goal_observations_validation, axis=0)

        # observations = {
        #     key: np.concatenate((
        #         sample_observations[key],
        #         goal_observations[key],
        #         goal_observations_validation[key]
        #     ), axis=0)
        #     for key in sample_observations.keys()
        # }
        # labels = np.concatenate((
        #     sample_labels, goal_labels, goal_validation_labels,
        # ), axis=0)
        # actions = np.concatenate((
        #     sample_actions, goal_actions, goal_validation_actions), axis=0)

        # (reward_observations,
        #  classifier_output,
        #  log_pi,
        #  discriminator_output,
        #  classifier_loss) = self._session.run(
        #     (self._reward_t,
        #      self._classifier_log_p_t,
        #      self._log_pi_t,
        #      self._discriminator_output_t,
        #      self._classifier_loss_t),
        #     feed_dict={
        #         **{
        #             self._placeholders['observations'][key]: values
        #             for key, values in sample_observations.items()
        #         },
        #         self._placeholders['labels']: sample_labels,
        #         self._placeholders['actions']: sample_actions,
        #     }
        # )

        (reward_negative_observations,
         # classifier_output_negative,
         # log_pi_negative,
         # discriminator_output_negative,
         negative_classifier_loss) = self._session.run(
            (self._reward_t,
             # self._classifier_log_p_t,
             # self._log_pi_t,
             # self._discriminator_output_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in sample_observations.items()
                },
                self._placeholders['labels']: sample_labels,
                # self._placeholders['actions']: sample_actions,
            }
        )

        (reward_goal_observations_training,
         # classifier_output_goal_training,
         # discriminator_output_goal_training,
         goal_classifier_training_loss) = self._session.run(
            (self._reward_t,
             # self._classifier_log_p_t,
             # self._discriminator_output_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in goal_observations.items()
                },
                self._placeholders['labels']: goal_labels
            }
        )

        (reward_goal_observations_validation,
         # classifier_output_goal_validation,
         # discriminator_output_goal_validation,
         goal_classifier_validation_loss) = self._session.run(
            (self._reward_t,
             # self._classifier_log_p_t,
             # self._discriminator_output_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in goal_observations_validation.items()
                },
                self._placeholders['labels']: goal_validation_labels
            }
        )

        diagnostics.update({
            # classifier loss averaged across the actual training batches
            'reward_learning/classifier_training_loss': np.mean(
                self._training_loss),
            # classifier loss sampling from the goal image pool
            'reward_learning/classifier_loss_sample_goal_obs_training': np.mean(
                goal_classifier_training_loss),
            'reward_learning/classifier_loss_sample_goal_obs_validation': np.mean(
                goal_classifier_validation_loss),
            'reward_learning/classifier_loss_sample_negative_obs': np.mean(
                negative_classifier_loss),
            'reward_learning/reward_negative_obs_mean': np.mean(
                reward_negative_observations),
            'reward_learning/reward_goal_obs_training_mean': np.mean(
                reward_goal_observations_training),
            'reward_learning/reward_goal_obs_validation_mean': np.mean(
                reward_goal_observations_validation),
            # 'reward_learning/classifier_negative_obs_log_p_mean': np.mean(
            #     classifier_output_negative),
            # 'reward_learning/classifier_goal_obs_training_log_p_mean': np.mean(
            #     classifier_output_goal_training),
            # 'reward_learning/classifier_goal_obs_validation_log_p_mean': np.mean(
            #     classifier_output_goal_validation),
            # 'reward_learning/discriminator_output_negative_mean': np.mean(
            #     discriminator_output_negative),
            # 'reward_learning/discriminator_output_goal_obs_training_mean': np.mean(
            #     discriminator_output_goal_training),
            # 'reward_learning/discriminator_output_goal_obs_validation_mean': np.mean(
            #     discriminator_output_goal_validation),

            # TODO: Figure out why converting to probabilities isn't working
            # 'reward_learning/classifier_negative_obs_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_negative_observations)),
            # 'reward_learning/classifier_goal_obs_training_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_goal_observations_training)),
            # 'reward_learning/classifier_goal_obs_validation_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_goal_observations_validation)),
        })

        return diagnostics
