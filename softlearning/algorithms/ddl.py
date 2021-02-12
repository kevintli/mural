import numpy as np
import tensorflow as tf

from .sac import SAC
from softlearning.models.utils import flatten_input_structure
from flatten_dict import flatten


class DDL(SAC):
    def __init__(
        self,
        distance_fn,
        goal_state,
        train_distance_fn_every_n_steps=64,
        use_ground_truth_distances=False,
        ddl_lr=3e-4,
        ddl_symmetric=False,
        ddl_clip_length=None,
        ddl_train_steps=1,
        ddl_batch_size=256,
        **kwargs,
    ):
        self._distance_fn = distance_fn
        if hasattr(self._distance_fn, 'classifier_params') and self._distance_fn.classifier_params is not None:
            self._ddl_use_classification = True
            self._ddl_max_distance = self._distance_fn.classifier_params['max_distance']
            self._ddl_bins = self._distance_fn.classifier_params['bins']
        else:
            self._ddl_use_classification = False

        # TODO: Make a goal proposer
        self._goal_state = goal_state

        self._train_distance_fn_every_n_steps = train_distance_fn_every_n_steps
        self._use_ground_truth_distances = use_ground_truth_distances
        self._ddl_lr = ddl_lr
        self._ddl_symmetric = ddl_symmetric
        self._ddl_clip_length = ddl_clip_length
        self._ddl_train_steps = ddl_train_steps
        self._ddl_batch_size = ddl_batch_size

        super(DDL, self).__init__(**kwargs)

    def _build(self):
        super(DDL, self)._build()
        self._init_ddl_update()

    def _init_placeholders(self):
        super(DDL, self)._init_placeholders()
        self._init_ddl_placeholders()

    def _init_ddl_placeholders(self):
        self._placeholders.update({
            's1': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f's1/{name}')
                for name, observation_space
                in self._training_environment.observation_space.spaces.items()
            },
            's2': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f's2/{name}')
                for name, observation_space
                in self._training_environment.observation_space.spaces.items()
            },
            'distances': tf.compat.v1.placeholder(
                dtype=np.float32, shape=(None, 1), name='distances')
        })

    def _init_ddl_update(self):
        distance_fn_inputs = self._distance_fn_inputs(
            s1=self._placeholders['s1'], s2=self._placeholders['s2'])
        distance_preds = self._distance_preds = (
            self._distance_fn(distance_fn_inputs))

        distance_targets = self._placeholders['distances']

        if self._ddl_use_classification:
            # Convert numerical distances into bins
            distance_labels = tf.cast(
                tf.squeeze(tf.clip_by_value(
                     distance_targets // (self._ddl_max_distance // self._ddl_bins), 
                     0, self._ddl_bins
                ), axis=1),
                tf.int32
            )
           
            distance_loss = self._distance_loss = (
                tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=distance_labels,
                    logits=distance_preds)
            )
        else:
            distance_loss = self._distance_loss = (
                tf.compat.v1.losses.mean_squared_error(
                    labels=distance_targets,
                    predictions=distance_preds,
                    weights=0.5)
            )

        ddl_optimizer = self._ddl_optimizer = (
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._ddl_lr,
                name='ddl_optimizer'))
        self._ddl_train_op = ddl_optimizer.minimize(
            loss=distance_loss,
            var_list=self._distance_fn.trainable_variables)

    def _get_ddl_feed_dict(self):
        # Sample pairs of points randomly within the same trajectories
        s1_indices = self.sampler.pool.random_indices(self._ddl_batch_size)
        max_path_length = self.sampler.max_path_length

        if self._ddl_symmetric:
            # Sample s1, s2 randomly within a trajectory (either one can come first)
            low = s1_indices // max_path_length * max_path_length
            high = (s1_indices // max_path_length + 1) * max_path_length
        else:
            # s2 must come *after* s1 in the trajectory
            low = s1_indices
            high = (s1_indices // max_path_length + 1) * max_path_length
        
        if self._ddl_clip_length:
            low = np.max([low, s1_indices - self._ddl_clip_length], axis=0)
            high = np.min([high, s1_indices + self._ddl_clip_length], axis=0)

        s2_indices = np.random.randint(low, high)

        s1 = self.sampler.pool.batch_by_indices(s1_indices)
        s2 = self.sampler.pool.batch_by_indices(s2_indices)

        if self._use_ground_truth_distances:
            # Compute ground truth distances to use for training
            s1_pos, s2_pos = s1['observations']['state_observation'], s2['observations']['state_observation']
            distances = self._training_environment._env.unwrapped._medium_maze_manhattan_distance(s1_pos, s2_pos)[:,None]
        else:
            distances = np.abs(s1_indices - s2_indices).astype(np.float32)[:,None]

        feed_dict = {
            **{
                self._placeholders['s1'][key]: s1['observations'][key]
                for key in self._distance_fn.observation_keys
            },
            **{
                self._placeholders['s2'][key]: s2['observations'][key]
                for key in self._distance_fn.observation_keys
            },
            self._placeholders['distances']: distances,
        }
        return feed_dict

    def _distance_fn_inputs(self, s1, s2):
        inputs_1 = {
            name: s1[name]
            for name in self._distance_fn.observation_keys
        }
        inputs_2 = {
            name: s2[name]
            for name in self._distance_fn.observation_keys
        }
        inputs = {
            's1': inputs_1,
            's2': inputs_2,
        }
        return flatten_input_structure(inputs)

    def _policy_inputs(self, observations):
        policy_inputs = flatten_input_structure({
            name: observations[name]
            for name in self._policy.observation_keys
        })
        return policy_inputs

    def _Q_inputs(self, observations, actions):
        Q_observations = {
            name: observations[name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure(
            {**Q_observations, 'actions': actions})
        return Q_inputs

    def _init_extrinsic_reward(self):
        """
        Initializes the DDL reward as -(distance to goal)
        The feed dict should set one of the s1/s2 placeholders the goal
        """
        distance_fn_inputs = self._distance_fn_inputs(
            s1=self._placeholders['s1'], s2=self._placeholders['s2'])
        distances_to_goal = self._distance_fn(distance_fn_inputs)
        if self._ddl_use_classification:
            distances_to_goal = tf.cast(tf.argmax(distances_to_goal, axis=-1), tf.float32)[:,None] * (self._ddl_max_distance // self._ddl_bins)
        self._unscaled_ext_reward = -distances_to_goal

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(DDL, self)._get_feed_dict(iteration, batch)
        placeholders_flat = flatten(self._placeholders)

        # === Set s1, s2 for training Qs ===
        if self._goal_state:
            feed_dict.update({
                self._placeholders['s1'][key]:
                feed_dict[self._placeholders['observations'][key]]
                for key in self._distance_fn.observation_keys
            })

            batch_size = feed_dict[next(iter(feed_dict))].shape[0]
            feed_dict.update({
                self._placeholders['s2'][key]:
                np.repeat(self._goal_state[key][None], batch_size, axis=0)
                for key in self._distance_fn.observation_keys
            })

        return feed_dict

    def _do_training(self, iteration, batch):
        super()._do_training(iteration, batch)
        if iteration % self._train_distance_fn_every_n_steps == 0:
            for _ in range(self._ddl_train_steps):
                ddl_feed_dict = self._get_ddl_feed_dict()
                self._session.run(self._ddl_train_op, ddl_feed_dict)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(DDL, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        ddl_feed_dict = self._get_ddl_feed_dict()
        overall_distance_loss = self._session.run(
            self._distance_loss,
            feed_dict=ddl_feed_dict)

        diagnostics.update({
            'ddl/overall_distance_loss': np.mean(overall_distance_loss),
        })

        if self._goal_state:
            goal_feed_dict = self._get_feed_dict(iteration, batch)
            goal_relative_distance_preds = self._session.run(
                self._distance_preds,
                feed_dict=goal_feed_dict)
            diagnostics.update({
                'ddl/goal_relative_distance_preds': np.mean(goal_relative_distance_preds),
            })

        return diagnostics


class DynamicsAwareEmbeddingDDL(DDL):
    def __init__(
            self,
            distance_fn,
            *args,
            normalize_distance_targets=False,
            use_l2_distance_targets=False,
            use_separate_embeddings=False,
            **kwargs):

        self._embedding_fn = distance_fn
        self._normalize_distance_targets = normalize_distance_targets
        self._use_l2_distance_targets = use_l2_distance_targets
        self._use_separate_embeddings = use_separate_embeddings

        if self._use_separate_embeddings:
            self._embedding_fns = [
                distance_fn,
                tf.keras.clone_model(distance_fn)
            ]
        else:
            self._embedding_fns = [distance_fn, distance_fn]

        super(DynamicsAwareEmbeddingDDL, self).__init__(distance_fn, *args, **kwargs)

    def _embedding_fn_inputs(self, s):
        return flatten_input_structure({
            name: s[name]
            for name in self._embedding_fn.observation_keys
        })

    def _init_ddl_update(self):
        s1_input, s2_input = (
            self._embedding_fn_inputs(self._placeholders['s1']),
            self._embedding_fn_inputs(self._placeholders['s2'])
        )
        phi_s1, phi_s2 = (
            self._embedding_fns[0](s1_input),
            self._embedding_fns[1](s2_input),
        )

        # Want the L2 norm in the embedding space to match dynamical distance
        # phi(s2) - phi(s1): batch_size x embedding_dim
        distance_preds = self._distance_preds = (
            tf.reduce_sum(tf.square(phi_s2 - phi_s1), axis=-1, keepdims=True))
        distance_targets = self._placeholders['distances']

        if self._normalize_distance_targets:
            # Distance targets are proportional to the squared norm in embedding space
            # so dividing by a constant divides by the square of that constant in the embedding
            distance_targets = distance_targets / np.sqrt(self.sampler.max_path_length)

        if self._use_l2_distance_targets:
            # Use l2 distance between s1 and s2 observations as the distance targets
            distance_targets = tf.zeros_like(distance_targets)
            # Aggregate distances between all observation keys
            for key in self._placeholders['s1']:
                distance_targets += tf.norm(
                    self._placeholders['s2'][key] - self._placeholders['s1'][key],
                    axis=-1, keepdims=True)

        distance_loss = self._distance_loss = (
            tf.compat.v1.losses.mean_squared_error(
                labels=distance_targets,
                predictions=distance_preds,
                weights=0.5)
        )

        ddl_optimizer = self._ddl_optimizer = (
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._ddl_lr,
                name='embedding_fn_optimizer'))

        var_list = (
            self._embedding_fn.trainable_variables
            if not self._use_separate_embeddings
            else (self._embedding_fns[0].trainable_variables
                + self._embedding_fns[1].trainable_variables))

        self._ddl_train_op = ddl_optimizer.minimize(
            loss=distance_loss,
            var_list=var_list)

    def _init_extrinsic_reward(self):
        """
        Initializes the DDL reward as -|phi(s) - phi(g)|
        The feed dict should set one of the s1/s2 placeholders the goal
        """
        s1_input, s2_input = (
            self._embedding_fn_inputs(self._placeholders['s1']),
            self._embedding_fn_inputs(self._placeholders['s2'])
        )
        phi_s1, phi_s2 = (
            self._embedding_fns[0](s1_input),
            self._embedding_fns[1](s2_input),
        )
        distances_to_goal = tf.reduce_sum(tf.square(phi_s2 - phi_s1), axis=-1, keepdims=True)
        self._unscaled_ext_reward = -distances_to_goal


from .vice import VICE
class DynamicsAwareEmbeddingVICE(VICE, DynamicsAwareEmbeddingDDL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def _epoch_before_hook(self, *args, **kwargs):
    #     super()._epoch_before_hook(*args, **kwargs)
    #     import ipdb; ipdb.set_trace()

    # def _get_feed_dict(self, *args):
    #     super(VICE, self)._get_feed_dict(*args)
