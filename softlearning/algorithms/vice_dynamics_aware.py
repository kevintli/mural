import numpy as np
import tensorflow as tf

from .sac import td_target
from .sac_classifier import SACClassifier
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class VICEDynamicsAware(SACClassifier):

    def __init__(self,
                 dynamics_model,
                 *args,
                 train_dynamics_model_every_n_steps=64,
                 dynamics_model_lr=3e-4,
                 dynamics_model_batch_size=256,
                 **kwargs):
        self._dynamics_model = dynamics_model
        self._dynamics_model_lr = dynamics_model_lr
        self._dynamics_model_batch_size = dynamics_model_batch_size
        self._train_dynamics_model_every_n_steps = train_dynamics_model_every_n_steps

        super(VICEDynamicsAware, self).__init__(*args, **kwargs)

        assert 'actions' in self._goal_examples and 'observations' in self._goal_examples

    def _build(self):
        super(VICEDynamicsAware, self)._build()
        self._init_dynamics_model_update()

    def _dynamics_model_inputs(self, observations, actions):
        dynamics_model_observations = {
            name: observations[name]
            for name in self._dynamics_model.observation_keys
        }
        dynamics_model_inputs = flatten_input_structure(
            {**dynamics_model_observations, 'actions': actions})
        return dynamics_model_inputs

    def _classifier_inputs(self, observations, actions):
        classifier_observations = {
            name: observations[name]
            for name in self._policy.observation_keys
        }
        dynamics_model_inputs = self._dynamics_model_inputs(observations, actions)
        dynamics_features = self._dynamics_model.encoder(dynamics_model_inputs)

        classifier_inputs = flatten_input_structure({
            **classifier_observations,
            'dynamics_features': dynamics_features,
            # 'actions': actions
        })
        return classifier_inputs

    def _init_dynamics_model_update(self):
        dynamics_model_inputs = self._dynamics_model_inputs(
            self._placeholders['observations'], self._placeholders['actions'])
        next_obs_preds = self._dynamics_model(dynamics_model_inputs)

        # First apply preprocessors then concatenate?
        next_obs_tensors = list(self._placeholders['next_observations'].values())
        next_obs_targets = tf.concat(next_obs_tensors, axis=0, name='next_obs_targets')

        loss = self._dynamics_model_loss = (
            tf.compat.v1.losses.mean_squared_error(
                labels=next_obs_targets,
                predictions=next_obs_preds,
                weights=0.5)
        )

        dynamics_model_optimizer = self._dynamics_model_optimizer = (
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._dynamics_model_lr,
                name='dynamics_model_optimizer'))
        self._dynamics_model_training_op = (
            dynamics_model_optimizer.minimize(
                loss=loss,
                var_list=self._dynamics_model.trainable_variables))

    def _init_extrinsic_reward(self):
        classifier_inputs = self._classifier_inputs(
            self._placeholders['observations'], self._placeholders['actions'])
        observation_logits = self._classifier(classifier_inputs)

        if self._reward_type == 'logits':
            self._reward_t = observation_logits
        elif self._reward_type == 'probabilities':
            self._reward_t = tf.nn.sigmoid(observation_logits)
        else:
            raise NotImplementedError(
                f"Unknown reward type: {self._reward_type}")

        self._unscaled_ext_reward = self._reward_t

    def _get_classifier_feed_dict(self):
        negatives_batch = self.sampler.random_batch(
            self._classifier_batch_size
        )

        negatives_obs = negatives_batch['observations']
        negatives_act = negatives_batch['actions']

        _key = next(iter(self._goal_examples['observations']))
        rand_positive_ind = np.random.randint(
            self._goal_examples['observations'][_key].shape[0],
            size=self._classifier_batch_size)
        positives_obs = {
            key: values[rand_positive_ind]
            for key, values in self._goal_examples['observations'].items()
        }
        positives_act = self._goal_examples['actions'][rand_positive_ind]

        labels_batch = np.zeros(
            (2 * self._classifier_batch_size, 2),
            dtype=np.int32)
        labels_batch[:self._classifier_batch_size, 0] = 1
        labels_batch[self._classifier_batch_size:, 1] = 1

        observations_batch = {
            key: np.concatenate((negatives_obs[key], positives_obs[key]), axis=0)
            for key in self._policy.observation_keys
        }
        actions_batch = np.concatenate([negatives_act, positives_act], axis=0)

        if self._mixup_alpha > 0:
            observations_batch, labels_batch, permutation_idx = (
                mixup(observations_batch,
                      labels_batch,
                      alpha=self._mixup_alpha,
                      return_permutation=True))
            actions_batch = actions_batch[permutation_idx]

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observations_batch[key]
                for key in self._policy.observation_keys
            },
            self._placeholders['actions']: actions_batch,
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

    def _init_classifier_update(self):
        classifier_inputs = self._classifier_inputs(
            self._placeholders['observations'], self._placeholders['actions'])
        # classifier_inputs = flatten_input_structure({
        #     name: self._placeholders['observations'][name]
        #     for name in self._classifier.observation_keys
        # })
        log_p = self._classifier(classifier_inputs)
        # policy_inputs = flatten_input_structure({
        #     name: self._placeholders['observations'][name]
        #     for name in self._policy.observation_keys
        # })
        policy_inputs = self._policy_inputs(self._placeholders['observations'])
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

    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)

    def _do_training(self, iteration, batch):
        super(VICEDynamicsAware, self)._do_training(iteration, batch)
        if iteration % self._train_dynamics_model_every_n_steps == 0:
            feed_dict = self._get_feed_dict(iteration, batch)
            self._session.run(self._dynamics_model_training_op, feed_dict)

    def get_reward(self, observations):
        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: observations[name]
                for name in self._policy.observation_keys
            }
        )
        return learned_reward

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):

        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        sample_obs = batch['observations']
        _key = next(iter(sample_obs))
        sample_act = batch['actions']
        n_samples = sample_obs[_key].shape[0]
        sample_labels = np.repeat(((1, 0), ), n_samples, axis=0)

        # goal_index = np.random.randint(
        #     self._goal_examples[next(iter(self._goal_examples))].shape[0],
        #     size=sample_observations[next(iter(sample_observations))].shape[0])
        # goal_observations = {
        #     key: values[goal_index] for key, values in self._goal_examples.items()
        #     if key in self._policy.observation_keys
        # }
        # goal_actions = self._goal_examples['actions'][goal_index]

        rand_idx_train = np.random.randint(
            self._goal_examples['observations'][_key].shape[0],
            size=n_samples)
        goal_train_obs = {
            key: values[rand_idx_train]
            for key, values in self._goal_examples['observations'].items()
        }
        goal_train_act = self._goal_examples['actions'][rand_idx_train]
        goal_train_labels = np.repeat(((0, 1), ), n_samples, axis=0)

        rand_idx_valid = np.random.randint(
            self._goal_examples_validation['observations'][_key].shape[0],
            size=n_samples)
        goal_valid_obs = {
            key: values[rand_idx_valid]
            for key, values in self._goal_examples_validation['observations'].items()
        }
        goal_valid_act = self._goal_examples_validation['actions'][rand_idx_valid]
        goal_valid_labels = np.repeat(((0, 1), ), n_samples, axis=0)

        (sample_rew,
         sample_loss) = self._session.run(
            (self._reward_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: value
                    for key, value in sample_obs.items()
                },
                self._placeholders['actions']: sample_act,
                self._placeholders['labels']: sample_labels,
            }
        )

        (goal_train_rew,
         goal_train_loss) = self._session.run(
            (self._reward_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: value
                    for key, value in goal_train_obs.items()
                },
                self._placeholders['actions']: goal_train_act,
                self._placeholders['labels']: goal_train_labels,
            }
        )

        (goal_valid_rew,
         goal_valid_loss) = self._session.run(
            (self._reward_t,
             self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in goal_valid_obs.items()
                },
                self._placeholders['actions']: goal_valid_act,
                self._placeholders['labels']: goal_valid_labels
            }
        )

        dynamics_model_loss = self._session.run(
            self._dynamics_model_loss,
            feed_dict=self._get_feed_dict(iteration, batch))

        diagnostics.update({
            'dynamics_model/training_loss': dynamics_model_loss,
            # classifier loss averaged across the actual training batches
            'vice/classifier_training_loss': np.mean(self._training_loss),
            # classifier loss sampling from the goal image pool
            'vice/classifier_loss_sample_goal_obs_training': np.mean(goal_train_loss),
            'vice/classifier_loss_sample_goal_obs_validation': np.mean(goal_valid_loss),
            'vice/classifier_loss_sample_negative_obs': np.mean(sample_loss),
            'vice/reward_negative_obs': np.mean(sample_rew),
            'vice/reward_goal_obs_training': np.mean(goal_train_rew),
            'vice/reward_goal_obs_validation': np.mean(goal_valid_rew),
        })

        # diagnostics = super(SACClassifier, self).get_diagnostics(
        #     iteration, batch, training_paths, evaluation_paths)
        # sample_observations = batch['observations']
        # sample_actions = batch['actions']
        # num_sample_observations = sample_observations[
        #     next(iter(sample_observations))].shape[0]
        # sample_labels = np.repeat(((1, 0), ), num_sample_observations, axis=0)

        # goal_index = np.random.randint(
        #     self._goal_examples[next(iter(self._goal_examples))].shape[0],
        #     size=sample_observations[next(iter(sample_observations))].shape[0])
        # goal_observations = {
        #     key: values[goal_index] for key, values in self._goal_examples.items()
        #     if key in self._policy.observation_keys
        # }
        # # Sample goal actions uniformly in action space
        # # action_space_dim = sample_actions.shape[1]
        # # goal_actions = np.random.uniform(
        # #     low=-1, high=1, size=(num_sample_observations, action_space_dim))
        # # goal_validation_actions = np.random.uniform(
        # #     low=-1, high=1, size=(num_sample_observations, action_space_dim))
        # goal_index_validation = np.random.randint(
        #     self._goal_examples_validation[
        #         next(iter(self._goal_examples_validation))].shape[0],
        #     size=sample_observations[next(iter(sample_observations))].shape[0])
        # goal_observations_validation = {
        #     key: values[goal_index_validation]
        #     for key, values in self._goal_examples_validation.items()
        #     if key in self._policy.observation_keys
        # }

        # num_goal_observations = goal_observations[
        #     next(iter(goal_observations))].shape[0]
        # goal_labels = np.repeat(((0, 1), ), num_goal_observations, axis=0)

        # num_goal_observations_validation = goal_observations_validation[
        #     next(iter(goal_observations_validation))].shape[0]
        # goal_validation_labels = np.repeat(
        #     ((0, 1), ), num_goal_observations_validation, axis=0)

        # (reward_negative_observations,
        #  negative_classifier_loss) = self._session.run(
        #     (self._reward_t,
        #      self._classifier_loss_t),
        #     feed_dict={
        #         **{
        #             self._placeholders['observations'][key]: values
        #             for key, values in sample_observations.items()
        #         },
        #         self._placeholders['labels']: sample_labels,
        #         # self._placeholders['actions']: sample_actions,
        #     }
        # )

        # (reward_goal_observations_training,
        #  goal_classifier_training_loss) = self._session.run(
        #     (self._reward_t,
        #      self._classifier_loss_t),
        #     feed_dict={
        #         **{
        #             self._placeholders['observations'][key]: values
        #             for key, values in goal_observations.items()
        #         },
        #         self._placeholders['labels']: goal_labels
        #     }
        # )

        # (reward_goal_observations_validation,
        #  goal_classifier_validation_loss) = self._session.run(
        #     (self._reward_t,
        #      self._classifier_loss_t),
        #     feed_dict={
        #         **{
        #             self._placeholders['observations'][key]: values
        #             for key, values in goal_observations_validation.items()
        #         },
        #         self._placeholders['labels']: goal_validation_labels
        #     }
        # )

        # diagnostics.update({
        #     # classifier loss averaged across the actual training batches
        #     'reward_learning/classifier_training_loss': np.mean(
        #         self._training_loss),
        #     # classifier loss sampling from the goal image pool
        #     'reward_learning/classifier_loss_sample_goal_obs_training': np.mean(
        #         goal_classifier_training_loss),
        #     'reward_learning/classifier_loss_sample_goal_obs_validation': np.mean(
        #         goal_classifier_validation_loss),
        #     'reward_learning/classifier_loss_sample_negative_obs': np.mean(
        #         negative_classifier_loss),
        #     'reward_learning/reward_negative_obs_mean': np.mean(
        #         reward_negative_observations),
        #     'reward_learning/reward_goal_obs_training_mean': np.mean(
        #         reward_goal_observations_training),
        #     'reward_learning/reward_goal_obs_validation_mean': np.mean(
        #         reward_goal_observations_validation),
        # })

        return diagnostics
