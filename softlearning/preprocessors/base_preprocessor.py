import abc

import numpy as np


class BasePreprocessor(object):
    """Abstract definition of observations preprocessor."""

    def __init__(self, observation_space, output_size):
        self._observation_space = observation_space
        self._output_size = output_size

    @property
    def input_shape(self):
        return self._observation_space.shape

    @property
    def output_shape(self):
        return (self._output_size, )

    @property
    def input_size(self):
        return int(np.product(self.input_shape))

    @property
    def output_size(self):
        return self._output_size

    @property
    def observation_space(self):
        return self._observation_space

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """Return preprocessed observation."""
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def decode(self, *args, **kwargs):
        raise NotImplementedError
