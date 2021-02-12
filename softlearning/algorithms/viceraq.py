import numpy as np

from .vice import VICE


class VICERAQ(VICE):
    def __init__(self,
                 active_query_frequency=5,
                 **kwargs):
        super(VICERAQ, self).__init__(**kwargs)
        self._active_query_frequency = active_query_frequency

    def _epoch_after_hook(self, *args, **kwargs):
        # TODO: this code is repeated from RAQ
        # figure out some clean way to reuse it

        if self._epoch % self._active_query_frequency == 0:
            batch_of_interest = self._pool.last_n_batch(
                self._epoch_length * self._active_query_frequency)
            observations_of_interest = batch_of_interest['observations']
            labels_of_interest = batch_of_interest['is_goals']

            rewards_of_interest = self._session.run(
                self._reward_t,
                feed_dict={
                    self._placeholders['observations'][name]:
                    observations_of_interest[name]
                    for name in self._classifier.observation_keys
                }
            )
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

            # TODO: Figure out if it makes sense to use these
            # "hard" negatives in some interesting way
            # else:
            #     self._negative_examples = np.concatenate([
            #             self._negative_examples,
            #             np.expand_dims(observations_of_interest[max_ind], axis=0)
            #             ])

        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(VICERAQ, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        diagnostics.update({
            'active_learning/positives-set-size': self._goal_examples[
                next(iter(self._goal_examples.keys()))].shape[0],
            # 'active_learning/negatives-set-size': self._negative_examples.shape[0],
        })

        return diagnostics
