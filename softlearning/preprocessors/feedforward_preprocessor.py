from gym import spaces

from softlearning.models.feedforward import feedforward_model

from .base_preprocessor import BasePreprocessor


class FeedforwardPreprocessor(BasePreprocessor):
    def __init__(self, observation_space, output_size, *args, **kwargs):
        super(FeedforwardPreprocessor, self).__init__(
            observation_space, output_size)

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self._feedforward = feedforward_model(
            *args,
            input_shapes=input_shapes,
            output_size=output_size,
            **kwargs)

    def transform(self, observation):
        transformed = self._feedforward(observation)
        return transformed
