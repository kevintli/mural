import numpy as np
import tensorflow as tf

from .sac import SAC, td_target
from .sac_classifier_multigoal import SACClassifierMultiGoal
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class VICEGANMultiGoal(SACClassifierMultiGoal):
    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dicts = self._get_classifier_feed_dicts()
            self._train_classifier_step(feed_dicts)


class VICEMultiGoal(SACClassifierMultiGoal):
    # TODO: Implement VICE multi-goal
    pass


class VICEGANTwoGoal(SAC):
    def __init__(
            self,
            classifier_0,
            classifier_1,
            goal_examples_0,
            goal_examples_1,
            goal_examples_validation_0,
            goal_examples_validation_1,
            classifier_lr=1e-4,
            classifier_batch_size=128,
            reward_type = 'logits',
            n_classifier_train_steps=5,
            classifier_optim_name='adam',
            mixup_alpha=0.2,
            **kwargs,
    ):
        # TODO: Pass in a list of classifiers, a list of goal examples/validations instead.
        self._classifier_0 = classifier_0
        self._goal_examples_0 = goal_examples_0
        self._goal_examples_validation_0 = goal_examples_validation_0
        self._classifier_1 = classifier_1
        self._goal_examples_1 = goal_examples_1
        self._goal_examples_validation_1 = goal_examples_validation_1

        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size
        self._mixup_alpha = mixup_alpha

        super(VICEGANTwoGoal, self).__init__(**kwargs)

    def _build(self):
        super(VICEGANTwoGoal, self)._build()
        self._init_classifier_update()

    def _init_placeholders(self):
        super(VICEGANTwoGoal, self)._init_placeholders()
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

        self._classifier_optimizer_0 = opt_func(
            learning_rate=self._classifier_lr,
            name='classifier_optimizer_0')
        self._classifier_optimizer_1 = opt_func(
            learning_rate=self._classifier_lr,
            name='classifier_optimizer_1')

        classifier_training_ops = [
            tf.contrib.layers.optimize_loss(
                self._classifier_loss_t_0,
                self.global_step,
                learning_rate=self._classifier_lr,
                optimizer=self._classifier_optimizer_0,
                variables=self._classifier_0.trainable_variables,
                increment_global_step=False,
            ),
            tf.contrib.layers.optimize_loss(
                self._classifier_loss_t_1,
                self.global_step,
                learning_rate=self._classifier_lr,
                optimizer=self._classifier_optimizer_1,
                variables=self._classifier_1.trainable_variables,
                increment_global_step=False,
            ),
       ]

        return classifier_training_ops

    def _init_classifier_update(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier_0.observation_keys
        })
        logits_0 = self._classifier_0(classifier_inputs)
        logits_1 = self._classifier_1(classifier_inputs)
        self._classifier_loss_t_0 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_0, labels=self._placeholders['labels']))
        self._classifier_loss_t_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_1, labels=self._placeholders['labels']))
        self._classifier_training_ops = self._get_classifier_training_ops()

    def _get_classifier_feed_dicts(self):
        # Get 2x the number of negatives, filter out by goal index
        negatives = self.sampler.random_batch(
            2 * self._classifier_batch_size)['observations']

        negative_ind_0 = (negatives['goal_index'] == 0).flatten()
        negative_ind_1 = (negatives['goal_index'] == 1).flatten()
        n_negatives_0, n_negatives_1 = (
            np.sum(negative_ind_0.astype(int)), np.sum(negative_ind_1.astype(int)))
        negatives_0 = {
            key: values[negative_ind_0]
            for key, values in negatives.items()
        }
        negatives_1 = {
            key: values[negative_ind_1]
            for key, values in negatives.items()
        }

        # Get positives from different goal pools
        rand_positive_ind_0 = np.random.randint(
            self._goal_examples_0[next(iter(self._goal_examples_0))].shape[0],
            size=self._classifier_batch_size)
        positives_0 = {
            key: values[rand_positive_ind_0]
            for key, values in self._goal_examples_0.items()
        }

        rand_positive_ind_1 = np.random.randint(
            self._goal_examples_1[next(iter(self._goal_examples_1))].shape[0],
            size=self._classifier_batch_size)
        positives_1 = {
            key: values[rand_positive_ind_1]
            for key, values in self._goal_examples_1.items()
        }

        # labels_batch_0 = np.zeros((2 * self._classifier_batch_size, 1))
        # labels_batch[self._classifier_batch_size:] = 1.0
        # labels_batch_0 = labels_batch.copy()
        # labels_batch_1 = labels_batch.copy()

        labels_batch_0 = np.zeros((n_negatives_0 + self._classifier_batch_size, 1))
        labels_batch_0[n_negatives_0:] = 1.0
        labels_batch_1 = np.zeros((n_negatives_1 + self._classifier_batch_size, 1))
        labels_batch_1[n_negatives_1:] = 1.0

        observations_batch_0 = {
            key: np.concatenate((negatives_0[key], positives_0[key]), axis=0)
            for key in self._classifier_0.observation_keys
        }
        observations_batch_1 = {
            key: np.concatenate((negatives_1[key], positives_1[key]), axis=0)
            for key in self._classifier_1.observation_keys
        }

        if self._mixup_alpha > 0:
            observation_batch_0, labels_batch_0 = mixup(
                observations_batch_0, labels_batch_0, alpha=self._mixup_alpha)
            observation_batch_1, labels_batch_1 = mixup(
                observations_batch_1, labels_batch_1, alpha=self._mixup_alpha)

        feed_dict_0 = {
            **{
                self._placeholders['observations'][key]:
                observations_batch_0[key]
                for key in self._classifier_0.observation_keys
            },
            self._placeholders['labels']: labels_batch_0
        }

        feed_dict_1 = {
            **{
                self._placeholders['observations'][key]:
                observations_batch_1[key]
                for key in self._classifier_1.observation_keys
            },
            self._placeholders['labels']: labels_batch_1
        }

        return feed_dict_0, feed_dict_1


    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        # TODO: pass through both, and filter by goal_index
        classifier_0_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier_0.observation_keys
        })
        classifier_1_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier_1.observation_keys
        })

        observation_logits_0 = self._classifier_0(classifier_0_inputs)
        observation_logits_1 = self._classifier_1(classifier_1_inputs)

        # TODO: Merge the two outputs, based on the info/obs/current_goal
        goal_index_mask = self._placeholders['observations']['goal_index']
        # Use above to merge the two.

        # Use observation_logits_1 where goal is 1, observation_logits_0 where goal is 0
        observation_logits = tf.where(
            tf.cast(goal_index_mask, dtype=tf.bool),
            x=observation_logits_1,
            y=observation_logits_0)

        self._reward_t = observation_logits

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return Q_target

    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dicts = self._get_classifier_feed_dicts()
            self._train_classifier_step(feed_dicts)

    def _train_classifier_step(self, feed_dicts):
        feed_dict_0, feed_dict_1 = feed_dicts
        classifier_training_op_0, classifier_training_op_1 = self._classifier_training_ops
        _, loss_0 = self._session.run((
                classifier_training_op_0, self._classifier_loss_t_0
            ), feed_dict_0)
        _, loss_1 = self._session.run((
                classifier_training_op_1, self._classifier_loss_t_1
            ), feed_dict_1)
        return (loss_0, loss_1)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(VICEGANTwoGoal, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        sample_observations = batch['observations']
        goal_index_0 = np.random.randint(
            self._goal_examples_0[next(iter(self._goal_examples_0))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations_0 = {
            key: self._goal_examples_0[key][goal_index_0]
            for key in self._goal_examples_0.keys()
        }
        goal_index_1 = np.random.randint(
            self._goal_examples_1[next(iter(self._goal_examples_1))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations_1 = {
            key: self._goal_examples_1[key][goal_index_1]
            for key in self._goal_examples_1.keys()
        }

        goal_index_validation_0 = np.random.randint(
            self._goal_examples_validation_0[
                next(iter(self._goal_examples_0))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        #goal_observations_validation_0 = (
        #    self._goal_examples_validation_0[goal_index_validation_0])
        goal_observations_validation_0 = {
            key: self._goal_examples_validation_0[key][goal_index_validation_0]
            for key in self._goal_examples_validation_0.keys()
        }

        goal_index_validation_1 = np.random.randint(
            self._goal_examples_validation_1[
                next(iter(self._goal_examples_1))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        # goal_observations_validation_1 = (
        #     self._goal_examples_validation_1[goal_index_validation_1])
        goal_observations_validation_1 = {
            key: self._goal_examples_validation_1[key][goal_index_validation_1]
            for key in self._goal_examples_validation_1.keys()
        }

        reward_sample_goal_observations_0, classifier_loss_0 = self._session.run(
            (self._reward_t, self._classifier_loss_t_0),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: np.concatenate((
                        sample_observations[key],
                        goal_observations_0[key],
                        goal_observations_validation_0[key]
                    ), axis=0)
                    for key in self._classifier_0.observation_keys
                },
                self._placeholders['labels']: np.concatenate([
                    np.zeros((sample_observations[next(iter(sample_observations))].shape[0], 1)),
                    np.ones((goal_observations_0[next(iter(goal_observations_0))].shape[0], 1)),
                    np.ones((goal_observations_validation_0[next(iter(goal_observations_validation_0))].shape[0], 1)),
                ])
            }
        )

        reward_sample_goal_observations_1, classifier_loss_1 = self._session.run(
            (self._reward_t, self._classifier_loss_t_1),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: np.concatenate((
                        sample_observations[key],
                        goal_observations_1[key],
                        goal_observations_validation_1[key]
                    ), axis=0)
                    for key in self._classifier_0.observation_keys
                },
                self._placeholders['labels']: np.concatenate([
                    np.zeros((sample_observations[next(iter(sample_observations))].shape[0], 1)),
                    np.ones((goal_observations_1[next(iter(goal_observations_1))].shape[0], 1)),
                    np.ones((goal_observations_validation_1[next(iter(goal_observations_validation_1))].shape[0], 1)),
                ])
            }
        )

        # TODO: Make this clearer. Maybe just make all the vectors
        # the same size and specify number of splits
        (reward_sample_observations_0,
         reward_goal_observations_0,
         reward_goal_observations_validation_0) = np.split(
             reward_sample_goal_observations_0,
             (
                 sample_observations[next(iter(sample_observations))].shape[0],
                 sample_observations[next(iter(sample_observations))].shape[0] + goal_observations_0[next(iter(goal_observations_0))].shape[0]
             ),
             axis=0)

        (reward_sample_observations_1,
         reward_goal_observations_1,
         reward_goal_observations_validation_1) = np.split(
             reward_sample_goal_observations_1,
             (
                 sample_observations[next(iter(sample_observations))].shape[0],
                 sample_observations[next(iter(sample_observations))].shape[0] + goal_observations_1[next(iter(goal_observations_1))].shape[0]
             ),
             axis=0)

        # TODO: fix this so that classifier loss is split into train and val
        # currently the classifier loss printed is the mean
        # classifier_loss_train, classifier_loss_validation = np.split(
        #     classifier_loss,
        #     (sample_observations.shape[0]+goal_observations.shape[0],),
        #     axis=0)

        diagnostics.update({
            # 'reward_learning/classifier_loss_train':
            # np.mean(classifier_loss_train),
            # 'reward_learning/classifier_loss_validation':
            # np.mean(classifier_loss_validation),
            'reward_learning/classifier_loss_0': classifier_loss_0,
            'reward_learning/classifier_loss_1': classifier_loss_1,

            'reward_learning/reward_sample_obs_mean_0': np.mean(
                reward_sample_observations_0),
            'reward_learning/reward_goal_obs_mean_0': np.mean(
                reward_goal_observations_0),
            'reward_learning/reward_goal_obs_validation_mean_0': np.mean(
                reward_goal_observations_validation_0),

            'reward_learning/reward_sample_obs_mean_1': np.mean(
                reward_sample_observations_1),
            'reward_learning/reward_goal_obs_mean_1': np.mean(
                reward_goal_observations_1),
            'reward_learning/reward_goal_obs_validation_mean_1': np.mean(
                reward_goal_observations_validation_1),
        })

        return diagnostics

    def _evaluate_rollouts(self, episodes, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics = super(VICEGANTwoGoal, self)._evaluate_rollouts(
            episodes, env)

        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: np.concatenate([
                    episode['observations'][name]
                    for episode in episodes
                ])
                for name in self._classifier_0.observation_keys
            })

        diagnostics[f'reward_learning/reward-mean'] = np.mean(learned_reward)
        diagnostics[f'reward_learning/reward-min'] = np.min(learned_reward)
        diagnostics[f'reward_learning/reward-max'] = np.max(learned_reward)
        diagnostics[f'reward_learning/reward-std'] = np.std(learned_reward)

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = super(VICEGANTwoGoal, self).tf_saveables
        saveables.update({
            '_classifier_optimizer_0': self._classifier_optimizer_0,
            '_classifier_optimizer_1': self._classifier_optimizer_1,
        })

        return saveables
