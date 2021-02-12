import numpy as np
import tensorflow as tf

# from .sac import SAC, td_target
from .multi_sac import MultiSAC
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class MultiSACClassifier(MultiSAC):
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

        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size
        self._mixup_alpha = mixup_alpha
        self._goal_conditioned = goal_conditioned

        super().__init__(**kwargs)

    def _build(self):
        super()._build()
        self._init_classifier_updates()

    def _init_placeholders(self):
        super()._init_placeholders()
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

    def _init_classifier_updates(self):
        assert self._num_goals == len(self._classifiers)
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

    def _init_external_rewards(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifiers[0].observation_keys
        })

        observation_logits_per_classifier = [
            classifier(classifier_inputs) for classifier in self._classifiers]

        # # DEBUG
        # # self._observation_logits_per_classifier = observation_logits_per_classifier
        # goal_indices = self._placeholders['observations']['goal_index']
        # goal_index_masks = [
        #     tf.equal(goal_indices, goal)
        #     for goal in range(self._num_goals)
        # ]

        # DEBUG
        # self._goal_index_masks = goal_index_masks

        # Replace the correct classification logits for the repsective goals
        # observation_logits = observation_logits_per_classifier[0]
        # for goal in range(1, self._num_goals):
        #     observation_logits = tf.where(
        #        goal_index_masks[goal],
        #        x=observation_logits_per_classifier[goal],
        #        y=observation_logits
        #     )
        self._unscaled_ext_rewards = observation_logits_per_classifier

    def _get_classifier_feed_dict(self, i):
        # Get feed dict for classifier i
        negatives = self._samplers[i].random_batch(
            self._classifier_batch_size)['observations']

        goal_examples = self._goal_example_pools[i]
        rand_positive_ind = np.random.randint(
            goal_examples[next(iter(goal_examples))].shape[0],
            size=self._classifier_batch_size)
        positives = {
            key: values[rand_positive_ind]
            for key, values in goal_examples.items()
        }

        labels_batch = np.zeros((2*self._classifier_batch_size, 1))
        labels_batch[self._classifier_batch_size:] = 1.0

        observations_batch = {
            key: np.concatenate((negatives[key], positives[key]), axis=0)
            for key in self._classifiers[i].observation_keys
        }

        if self._mixup_alpha > 0:
            observation_batch, labels_batch = mixup(
                observations_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observations_batch[key]
                for key in self._classifiers[i].observation_keys
            },
            self._placeholders['labels']: labels_batch
        }

        return feed_dict

    def get_reward(self, observations):
        learned_reward = self._session.run(
            self._unscaled_ext_rewards[self._goal_index],
            feed_dict={
                self._placeholders['observations'][name]: observations[name]
                for name in self._policies[self._goal_index].observation_keys
                # for name in self._classifiers[0].observation_keys
            }
        )
        return learned_reward

    def _init_training(self):
        super()._init_training()
        for i in range(self._num_goals):
            try:
                self._samplers[i].set_algorithm(self)
            except:
                pass

    def _epoch_after_hook(self, *args, **kwargs):
        losses_per_classifier = [[] for _ in range(self._num_goals)]
        if self._epoch == 0: # Why epoch == 0?
            for i in range(self._num_goals):
                if self._ext_reward_coeffs[i]:
                    for _ in range(self._n_classifier_train_steps):
                        feed_dict = self._get_classifier_feed_dict(i)
                        losses_per_classifier[i].append(
                            self._train_classifier_step(i, feed_dict))
        self._training_losses_per_classifier = [
            np.concatenate(loss, axis=-1) if loss else np.array([]) for loss in losses_per_classifier]

    def _train_classifier_step(self, i, feed_dict):
        # Train classifier i on feed_dict
        _, loss = self._session.run((
            self._classifier_training_ops[i], self._classifier_losses_t
        ), feed_dict)
        return loss

    def get_diagnostics(self,
                        iteration,
                        batches,
                        training_paths_per_policy,
                        evaluation_paths_per_policy):
        diagnostics = super().get_diagnostics(
            iteration, batches, training_paths_per_policy, evaluation_paths_per_policy)

        for goal_index in range(self._num_goals):
            if self._ext_reward_coeffs[goal_index]:
                sample_obs = batches[goal_index]['observations']
                n_sample_obs = sample_obs[next(iter(sample_obs))].shape[0]

                training_goals = self._goal_example_pools[goal_index]
                validation_goals = self._goal_example_validation_pools[goal_index]
                rand_indices = np.random.randint(
                    training_goals[next(iter(training_goals))].shape[0],
                    size=n_sample_obs)
                rand_training_goals = {
                    key: training_goals[key][rand_indices]
                    for key in training_goals.keys()
                }
                rand_indices = np.random.randint(
                    validation_goals[next(iter(validation_goals))].shape[0],
                    size=n_sample_obs)
                rand_validation_goals = {
                    key: validation_goals[key][rand_indices]
                    for key in validation_goals.keys()
                }

                rewards, classifier_losses = self._session.run(
                    (self._unscaled_ext_rewards[goal_index],
                     self._classifier_losses_t[goal_index]),
                    feed_dict={
                        **{
                            self._placeholders[
                                'observations'][key]: np.concatenate((
                                    sample_obs[key],
                                    rand_training_goals[key],
                                    rand_validation_goals[key]
                                ), axis=0)
                            for key in self._classifiers[goal_index].observation_keys
                        },
                        self._placeholders['labels']: np.concatenate([
                            np.zeros((n_sample_obs, 1)),
                            np.ones((n_sample_obs, 1)),
                            np.ones((n_sample_obs, 1)),
                        ])
                    }
                )

                (sample_rewards,
                 goal_rewards,
                 goal_rewards_validation) = np.split(
                     rewards,
                    (n_sample_obs, n_sample_obs + n_sample_obs),
                    axis=0)

                diagnostics.update({
                    f'reward_learning/classifier_training_loss_{goal_index}':
                    self._training_losses_per_classifier[goal_index]
                })
                diagnostics.update({
                    f'reward_learning/sample_reward_mean_{goal_index}':
                    np.mean(sample_rewards)
                })
                diagnostics.update({
                    f'reward_learning/goal_reward_mean_{goal_index}':
                    np.mean(goal_rewards)
                })
                diagnostics.update({
                    f'reward_learning/goal_validation_reward_mean_{goal_index}':
                    np.mean(goal_rewards_validation)
                })

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = super().tf_saveables
        saveables.update({
            '_classifier_optimizer_' + str(goal): self._classifier_optimizers[goal]
            for goal in range(self._num_goals)
        })

        return saveables
