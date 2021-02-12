import numpy as np

from .vice_gan import VICEGAN
from .sac_classifier import SACClassifier
from softlearning.misc.utils import mixup


class VICEGANGoalConditioned(VICEGAN):
    def _get_classifier_feed_dict(self):
        import ipdb; ipdb.set_trace()
        
        negatives = self.sampler.random_batch(self._classifier_batch_size)['observations']
        #rand_positive_ind = np.random.randint(self._goal_examples.shape[0], size=self._classifier_batch_size)
        #positives = self._goal_examples[rand_positive_ind]
        
        state_goal_size = negatives[next(iter(negatives.keys()))].shape[1]
        
        #state_goal_size = negatives.shape[1]
        #assert state_goal_size%2 == 0, 'States and goals should be concatenated together, \
        #    so the total space has to be even'
        
        state_size = int(state_goal_size/2)
        #positives = np.concatenate([neg_observations[:, state_size:], neg_observations[:, state_size:]], axis=1)
        # this concatenates the goal examples from the environment
        positives = {
            key : np.concatenate(
                [negatives[key][:, :, :, 3:], negatives[key][:, :, :, 3:]], axis=3)
            for key in self._classifier.observation_keys
        }

        labels_batch = np.zeros((2 * self._classifier_batch_size, 1))
        labels_batch[self._classifier_batch_size:] = 1.0

        observation_batch = {
            key : np.concatenate((negatives[key], positives[key]), axis=0)
            for key in self._classifier.observation_keys
        }

        if self._mixup_alpha > 0:
            observation_batch, labels_batch = mixup(
                observation_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observation_batch[key]
                for key in self._classifier.observation_keys
            },
            self._placeholders['labels']: labels_batch
        }

        return feed_dict

    def _set_training_environment_image_goal(self, image):
        self._training_environment._env.wrapped_env.set_image_goal(image)

    def _timestep_before_hook(self):
        # if first step in epoch
        if self._timestep == 0 and np.random.random() < self._hindsight_goal_prob:
            new_goal = self._pool.random_batch(1, relabel=False)['observations']
            new_goal_observation = new_goal['observations']
            state_goal_size = new_goal_observation.shape[1]
            assert state_goal_size % 2 == 0, 'States and goals should be concatenated together, \
                so the total space has to be even' 
            state_size = int(state_goal_size/2)
            self._set_training_environment_image_goal(new_goal_observation[0, :state_size])

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        # TODO: figure out some classifier diagnostics that
        # don't involve a pre-defined validation set.

        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)
        return diagnostics
