"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np

from gym import spaces
from gym import ObservationWrapper

import skimage

STATE_KEY = 'state'

from skimage import transform

class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""
    def __init__(self,
                 env,
                 pixels_only=True,
                 render_kwargs=None,
                 normalize=False,
                 observation_key='pixels',
                 camera_ids=(-1,)):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            observation_key: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains the specified
                `observation_key`.
        """
        super(PixelObservationWrapper, self).__init__(env)
        if render_kwargs is None:
            render_kwargs = {}

        render_mode = render_kwargs.pop('mode', 'rgb_array')
        assert render_mode == 'rgb_array', render_mode
        render_kwargs['mode'] = 'rgb_array'

        # Specify number of cameras and their ids to render with
        # The observation will become a depthwise concatenation of all the
        # images gathered from these specified cameras.
        self._camera_ids = camera_ids
        self._render_kwargs_per_camera = [
            {
                'mode': 'rgb_array',
                'width': render_kwargs['width'],
                'height': render_kwargs['height'],
                'camera_id': camera_id,
            }
            for camera_id in self._camera_ids
        ]

        wrapped_observation_space = env.observation_space

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key
        if 'box_warp' in render_kwargs.keys():
            self._box_warp = render_kwargs.pop('box_warp')
        else:
            self._box_warp = False

        # import ipdb; ipdb.set_trace()
        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only and observation_key in invalid_keys:
            raise ValueError("Duplicate or reserved observation key {!r}."
                             .format(observation_key))

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        self._normalize = normalize

        # Extend observation space with pixels.
        pixels = self._get_pixels()

        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float('inf'), float('inf')) # Fix for normalized between [-1, 1]
        else:
            raise TypeError(pixels.dtype)

        pixels_space = spaces.Box(
            shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
        self.observation_space.spaces[observation_key] = pixels_space

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _get_pixels(self):
        pixels = []
        # Render rgb_array for each camera
        for render_kwargs in self._render_kwargs_per_camera:
            width = render_kwargs.get('width')
            height = render_kwargs.get('height')
            _pixels = self.env.render(**{**self._render_kwargs,
                                         'width': width, 'height': height})

            # TODO: Do this anti-aliasing in Mujoco render instead
            _pixels = skimage.transform.resize(
                _pixels, (width, height), anti_aliasing=True, preserve_range=True)

            if self._box_warp:
                # warp image
                scale_factor = width / 32

                if self._env.env._is_hardware:
                    rect = np.array([[5, 10], [27, 10], [0, 26], [31, 26]], np.float32) * scale_factor
                else:
                    rect = np.array([[3, 6], [28, 6], [0, 31], [31, 31]], np.float32) * scale_factor
                dst = np.array([[0, 0], [31, 0], [0, 31], [31, 31]], np.float32) * scale_factor
                tform3 = transform.ProjectiveTransform()
                tform3.estimate(dst, rect)

                _pixels = transform.warp(_pixels, tform3, output_shape=(width, height))

            _pixels = np.rint(_pixels).astype(np.uint8)

            pixels.append(_pixels)

            assert len(_pixels.shape) == 3, 'Invalid image shape, needs to be (W, H, D)'

        # Stack channel-wise
        pixels = np.dstack(pixels)

        if self._normalize:
            pixels = (2. / 255. * pixels) - 1.

        return pixels


    def _add_pixel_observation(self, observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(observation)(observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = observation

        pixels = self._get_pixels()
        observation[self._observation_key] = pixels

        return observation
