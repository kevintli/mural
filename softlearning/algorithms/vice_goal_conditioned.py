import numpy as np

from .vice import VICE
from .sac_classifier import SACClassifier
from softlearning.misc.utils import mixup


class VICEGoalConditioned(VICE):
    def _timestep_before_hook(self, *args, **kwargs):
        # TODO: implement goal setting, something like
        # goal = self.pool.get_goal...
        # self.env.set_goal(goal)
        return super(VICEGoalConditioned, self)._timestep_before_hook(
            *args, **kwargs)

    def _get_classifier_feed_dict(self):
        negatives = self.sampler.random_batch(
            self._classifier_batch_size)['observations']

        state_goal_size = negatives.shape[1]
        assert state_goal_size % 2 == 0, (
            "States and goals should be concatenated together,"
            " so the total space has to be even.")

        state_size = int(state_goal_size / 2)
        positives = np.concatenate((
            negatives[:, state_size:],
            negatives[:, state_size:]
        ), axis=1)

        labels_batch = np.zeros(
            (2 * self._classifier_batch_size, 2), dtype=np.int32)
        labels_batch[:self._classifier_batch_size, 0] = 1
        labels_batch[self._classifier_batch_size:, 1] = 1
        observation_batch = np.concatenate([negatives, positives], axis=0)

        if self._mixup_alpha > 0:
            observation_batch, labels_batch = mixup(
                observation_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            self._observations_ph: observation_batch,
            self._label_ph: labels_batch
        }

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        # TODO: figure out some classifier diagnostics that
        # don't involve a pre-defined validation set
        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)
        return diagnostics
