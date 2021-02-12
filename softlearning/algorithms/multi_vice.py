import numpy as np

from .multi_sac_classifier import MultiSACClassifier


class MultiVICEGAN(MultiSACClassifier):
    def _epoch_after_hook(self, *args, **kwargs):
        losses_per_classifier = [[] for _ in range(self._num_goals)]
        for i in range(self._num_goals):
            if self._ext_reward_coeffs[i]:
                for _ in range(self._n_classifier_train_steps):
                    feed_dict = self._get_classifier_feed_dict(i)
                    losses_per_classifier[i].append(
                        self._train_classifier_step(i, feed_dict))
        self._training_losses_per_classifier = [
            np.concatenate(loss, axis=-1)
            if loss else np.array([]) for loss in losses_per_classifier]
