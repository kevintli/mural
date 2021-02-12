from gym.spaces import Box, Dict
import numpy as np

from multiworld.core.wrapper_env import ProxyEnv


class FlatGoalEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=False,
    ):
        self.quick_init(locals())
        super(FlatGoalEnv, self).__init__(wrapped_env)

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']

        self._append_goal_to_obs = append_goal_to_obs
        # if append_goal_to_obs:
        #     obs_keys += goal_keys
        for k in obs_keys:
            assert k in self.wrapped_env.observation_space.spaces
        for k in goal_keys:
            assert k in self.wrapped_env.observation_space.spaces
        assert isinstance(self.wrapped_env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict

        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys

        self.observation_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in keys
            ]),
        )
        self.goal_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in goal_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in goal_keys
            ]),
        )
        self._goal = None

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys
        flat_obs = np.hstack([obs[k] for k in keys])
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        self._goal = np.hstack([obs[k] for k in self.goal_keys])
        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys
        flat_obs = np.hstack([obs[k] for k in keys])
        return flat_obs

    def get_goal(self):
        return self._goal


class FlatEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            use_robot_state=False,
            robot_state_dims=4,
    ):
        self.quick_init(locals())
        super(FlatEnv, self).__init__(wrapped_env)

        self.use_robot_state = use_robot_state
        img_dims = np.prod(self.image_shape)*3 # 3 channels
        if self.use_robot_state:
            total_dim = img_dims + robot_state_dims
        else:
            total_dim = img_dims
        self.observation_space = Box(low=-10.0, high=10.0, shape=(total_dim,))

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        flat_obs = obs['image'].flatten()/255.0
        if self.use_robot_state:
            flat_obs = np.concatenate((flat_obs, obs['state']))
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        flat_obs = obs['image'].flatten()/255.0
        if self.use_robot_state:
            flat_obs = np.concatenate((flat_obs, obs['state']))
        return flat_obs


class ImageDictWrapper(ProxyEnv):

    def __init__(self, env):
        self.quick_init(locals())
        super(ImageDictWrapper, self).__init__(env)

        self.wrapped_env.wrapped_env.transpose = True

        self.image_length = 48*48*3
        img_space = Box(0, 1, (self.image_length,), dtype=np.float32)
        spaces = {'image': img_space}
        self.observation_space = Dict(spaces)

    def step(self, action):
        obs, rew, done, info = self.wrapped_env.step(action)
        obs = dict(image=obs)
        return obs, rew, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        obs = dict(image=obs)
        return obs
