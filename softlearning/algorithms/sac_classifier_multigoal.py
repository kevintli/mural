import numpy as np
import tensorflow as tf

from .sac import SAC, td_target
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class SACClassifierMultiGoal(SAC):
    def __init__(
        self,
        classifiers,
        goal_example_pools,
        goal_example_validation_pools,
        classifier_lr=1e-4,
        classifier_batch_size=128,
        reward_type='logits',
        n_classifier_train_steps=int(1e4),
        classifier_optim_name='adam',
        mixup_alpha=0.2,
        goal_conditioned=False,
        **kwargs,
    ):
        self._classifiers = classifiers
        self._goal_example_pools = goal_example_pools
        self._goal_example_validation_pools = goal_example_validation_pools

        assert classifiers and len(classifiers) == len(goal_example_pools) \
            and len(classifiers) == len(goal_example_validation_pools), \
            'Number of goal classifiers must match the number of goal pools'

        self._num_goals = len(classifiers)

        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size
        self._mixup_alpha = mixup_alpha
        self._goal_conditioned = goal_conditioned

        super(SACClassifierMultiGoal, self).__init__(**kwargs)

    def _build(self):
        super(SACClassifierMultiGoal, self)._build()
        self._init_classifier_update()

    def _init_placeholders(self):
        super(SACClassifierMultiGoal, self)._init_placeholders()
        self._placeholders['labels'] = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='labels',
        )

    def _get_classifier_training_ops(self):
        if self._classifier_optim_name == 'adam':
            opt_func = tf.train.AdamOptimizer
        elif self._classifier_optim_name == 'sgd':
            opt_func = tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

        self._classifier_optimizers = [
            opt_func(
                learning_rate=self._classifier_lr,
                name='classifier_optimizer_' + str(goal)
            )
            for goal in range(self._num_goals)
        ]

        classifier_training_ops = [
            tf.contrib.layers.optimize_loss(
                classifier_loss_t,
                self.global_step,
                learning_rate=self._classifier_lr,
                optimizer=classifier_optimizer,
                variables=classifier.trainable_variables,
                increment_global_step=False,
            )
            for classifier_loss_t, classifier_optimizer, classifier
                in zip(self._classifier_losses_t,
                       self._classifier_optimizers,
                       self._classifiers)
        ]

        return classifier_training_ops

    def _init_classifier_update(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifiers[0].observation_keys
        })

        goal_logits = [classifier(classifier_inputs)
                       for classifier in self._classifiers]

        self._classifier_losses_t = [
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=self._placeholders['labels']))
            for logits in goal_logits
        ]

        self._classifier_training_ops = self._get_classifier_training_ops()

    def _init_external_reward(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifiers[0].observation_keys
        })

        observation_logits_per_classifier = [
            classifier(classifier_inputs) for classifier in self._classifiers]

        # DEBUG
        # self._observation_logits_per_classifier = observation_logits_per_classifier
        goal_indices = self._placeholders['observations']['goal_index']
        goal_index_masks = [
            tf.equal(goal_indices, goal)
            for goal in range(self._num_goals)
        ]

        # DEBUG
        # self._goal_index_masks = goal_index_masks

        # Replace the correct classification logits for the repsective goals
        observation_logits = observation_logits_per_classifier[0]
        for goal in range(1, self._num_goals):
            observation_logits = tf.where(
               goal_index_masks[goal],
               x=observation_logits_per_classifier[goal],
               y=observation_logits
            )

        self._ext_reward = self._reward_t = observation_logits

    def _get_classifier_feed_dicts(self):
        # Sample N x the normal amount of observations, where N is
        # the number of goals.
        negatives = self.sampler.random_batch(
            self._num_goals * self._classifier_batch_size)['observations']

        # Split up the sample observations based on the goal index.
        # TODO: Make it split based on the goal qpos
        negative_inds = [
            (negatives['goal_index'] == goal).flatten()
            for goal in range(self._num_goals)
        ]
        negatives_per_goal = [
            {
                key: values[negative_ind]
                for key, values in negatives.items()
            }
            for negative_ind in negative_inds
        ]

        # Get positives from different goal pools
        goal_example_pool_sizes = [
            goal_example_pool[next(iter(goal_example_pool.keys()))].shape[0]
            for goal_example_pool in self._goal_example_pools
        ]
        rand_positive_indices = [
            np.random.randint(
                goal_example_pool_size,
                size=self._classifier_batch_size)
            for goal_example_pool_size in goal_example_pool_sizes
        ]
        positives_per_goal = [
            {
                key: values[rand_positive_ind]
                for key, values in goal_examples.items()
            }
            for rand_positive_ind, goal_examples
                in zip(rand_positive_indices, self._goal_example_pools)
        ]

        labels_batches = []
        for goal in range(self._num_goals):
            n_negatives = np.sum(negative_inds[goal].astype(int))
            n_positives = self._classifier_batch_size
            labels_batch = np.concatenate([
                np.zeros((n_negatives, 1)),
                np.ones((n_positives, 1)),
            ])
            labels_batches.append(labels_batch)

        # labels_batch = np.zeros((2 * self._classifier_batch_size, 1))
        # labels_batch[self._classifier_batch_size:] = 1.0
        # labels_batches = [labels_batch.copy() for _ in range(self._num_goals)]

        observation_batches = [
            {
                key: np.concatenate((_negatives[key], _positives[key]), axis=0)
                for key in self._classifiers[0].observation_keys
            }
            for _negatives, _positives in zip(negatives_per_goal, positives_per_goal)
        ]

        if self._mixup_alpha > 0:
            for goal_index in range(self._num_goals):
                observation_batches[goal_index], labels_batches[goal_index] = mixup(
                    observation_batches[goal_index], labels_batches[goal_index], alpha=self._mixup_alpha)

        feed_dicts = [
            {
                **{
                    self._placeholders['observations'][key]:
                    observations_batch[key]
                    for key in self._classifiers[0].observation_keys
                },
                self._placeholders['labels']: labels_batch
            }
            for observations_batch, labels_batch in zip(observation_batches, labels_batches)
        ]

        return feed_dicts

    def _epoch_after_hook(self, *args, **kwargs):
        if self._epoch == 0:
            for i in range(self._n_classifier_train_steps):
                feed_dicts = self._get_classifier_feed_dicts()
                self._train_classifier_step(feed_dicts)

    def _train_classifier_step(self, feed_dicts):
        losses = []
        for feed_dict, classifier_training_op, classifier_loss_t \
            in zip(feed_dicts,
                   self._classifier_training_ops,
                   self._classifier_losses_t):
            _, loss = self._session.run((
                classifier_training_op, classifier_loss_t
                ), feed_dict)
            losses.append(loss)
        self._training_losses = losses
        return losses

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SACClassifierMultiGoal, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)
        sample_obs = batch['observations']
        n_sample_obs = sample_obs[next(iter(sample_obs))].shape[0]

        goal_indices = [
            np.random.randint(
                goal_examples[next(iter(goal_examples))].shape[0],
                size=n_sample_obs)
            for goal_examples in self._goal_example_pools
        ]
        goal_observations_per_goal = [
            {
                key: goal_examples[key][goal_index]
                for key in goal_examples.keys()
            }
            for goal_examples, goal_index in zip(self._goal_example_pools, goal_indices)
        ]
        goal_indices_validation = [
            np.random.randint(
                goal_examples_validation[next(iter(goal_examples_validation))].shape[0],
                size=n_sample_obs)
            for goal_examples_validation in self._goal_example_validation_pools
        ]
        goal_observations_validation_per_goal = [
            {
                key: goal_examples_validation[key][goal_index]
                for key in goal_examples_validation.keys()
            }
            for goal_examples_validation, goal_index in
                zip(self._goal_example_validation_pools, goal_indices_validation)
        ]

        reward_sample, reward_goal, reward_goal_validation, losses = [], [], [], []
        for goal_index in range(self._num_goals):
            goal_obs = goal_observations_per_goal[goal_index]
            n_goal_obs = goal_obs[next(iter(goal_obs))].shape[0]
            goal_obs_validation = goal_observations_validation_per_goal[goal_index]
            n_goal_obs_validation = goal_obs_validation[next(iter(goal_obs_validation))].shape[0]

            # DEBUG
            # observation_logits_0, observation_logits_1 = self._observation_logits_per_classifier
            # goal_index_mask_0, goal_index_mask_1 = self._goal_index_masks

            try:
                obs_feed_dict = {
                    self._placeholders['observations'][key]: np.concatenate((
                        sample_obs[key],
                        goal_obs[key],
                        goal_obs_validation[key]
                    ), axis=0)
                    for key in self._policy.observation_keys
                }
            except:
                obs_feed_dict = {
                    self._placeholders['observations'][key]: np.concatenate((
                        sample_obs[key],
                        goal_obs[key],
                        goal_obs_validation[key]
                    ), axis=0)
                    for key in self._classifiers[goal_index].observation_keys
                }

            reward_sample_goal_observations, classifier_loss = self._session.run(
                (self._reward_t, self._classifier_losses_t[goal_index]),
                feed_dict={
                    **obs_feed_dict,
                    # **{
                    #     self._placeholders['observations'][key]: np.concatenate((
                    #         sample_obs[key],
                    #         goal_obs[key],
                    #         goal_obs_validation[key]
                    #     ), axis=0)
                    #     for key in self._policy.observation_keys
                    #     # for key in self._classifiers[goal_index].observation_keys
                    # },
                    self._placeholders['labels']: np.concatenate([
                        np.zeros((n_sample_obs, 1)),
                        np.ones((n_goal_obs, 1)),
                        np.ones((n_goal_obs_validation, 1)),
                    ])
                }
            )
            (reward_sample_observations,
            reward_goal_observations,
            reward_goal_observations_validation) = np.split(
                reward_sample_goal_observations,
                (n_sample_obs, n_sample_obs + n_goal_obs),
                axis=0)
            reward_sample.append(reward_sample_observations)
            reward_goal.append(reward_goal_observations)
            reward_goal_validation.append(reward_goal_observations_validation)
            losses.append(classifier_loss)

        # Add losses/classifier outputs to the dictionary
        diagnostics.update({
            # 'reward_learning/classifier_training_loss_' + str(goal): losses[goal]
            'reward_learning/classifier_training_loss_' + str(goal):
            self._training_losses[goal]
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_sample_obs_mean_' + str(goal):
            np.mean(reward_sample[goal])
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_goal_obs_mean_' + str(goal):
            np.mean(reward_goal[goal])
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_goal_obs_validation_mean_' + str(goal):
            np.mean(reward_goal_validation[goal])
            for goal in range(self._num_goals)
        })

        return diagnostics

    def _evaluate_rollouts(self, episodes, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics = super(SACClassifierMultiGoal, self)._evaluate_rollouts(
            episodes, env)

        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: np.concatenate([
                    episode['observations'][name]
                    for episode in episodes
                ])
                for name in self._policy.observation_keys
                # for name in self._classifiers[0].observation_keys
            })

        diagnostics[f'reward_learning/reward-mean'] = np.mean(learned_reward)
        diagnostics[f'reward_learning/reward-min'] = np.min(learned_reward)
        diagnostics[f'reward_learning/reward-max'] = np.max(learned_reward)
        diagnostics[f'reward_learning/reward-std'] = np.std(learned_reward)

        return diagnostics

    def get_reward(self, observations):
        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: observations[name]
                for name in self._policy.observation_keys
                # for name in self._classifiers[0].observation_keys
            }
        )
        return learned_reward

    @property
    def tf_saveables(self):
        saveables = super(SACClassifierMultiGoal, self).tf_saveables
        saveables.update({
            '_classifier_optimizer_' + str(goal): self._classifier_optimizers[goal]
            for goal in range(self._num_goals)
        })

        return saveables
