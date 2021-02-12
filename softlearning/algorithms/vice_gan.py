import tensorflow as tf

from softlearning.models.utils import flatten_input_structure
from .sac import td_target
from .sac_classifier import SACClassifier


class VICEGAN(SACClassifier):
    """
    A modification on the VICE[1] algorithm which uses a simple discriminator
    (similar to generative adversarial networks).

    References
    ----------
    [1] Variational Inverse Control with Events: A General
    Framework for Data-Driven Reward Definition. Justin Fu, Avi Singh,
    Dibya Ghosh, Larry Yang, Sergey Levine, NIPS 2018.
    """
    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)
