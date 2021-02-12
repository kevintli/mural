import abc
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_asset_full_path
import os.path as osp
from multiworld.envs.env_util import get_asset_full_path

PRESET1 = np.array([
    [-3, 0],
    [0, -3],
])
DEFAULT_GOAL = [-2., -2., 0.565, 1., 0., 0., 0., 0.,
                1., 0., -1., 0., -1., 0., 1., -3.,
                -3., 0.75, 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0.]




class AntEnv(MujocoEnv, Serializable, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            reward_type='dense',
            count_bonus_coeff=None,
            n_bins=16,
            target_radius=0.5, # Required distance to goal when using sparse reward
            norm_order=2,
            frame_skip=5,
            two_frames=False,
            vel_in_state=True,
            ant_low=list([-6, -6]),
            ant_high=list([6, 6]),
            goal_low=list([-6, -6]),
            goal_high=list([6, 6]),
            model_path='classic_mujoco/normal_gear_ratio_ant.xml',
            use_low_gear_ratio=False,
            goal_is_xy=False,
            goal_is_qpos=False,
            init_qpos=None,
            fixed_goal=None,
            diagnostics_goal=None,
            init_xy_mode='corner',
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 0.9),
            # health_reward=10,
            done_penalty=0,
            goal_sampling_strategy='uniform',
            presampled_goal_paths='',
            *args,
            **kwargs):
        assert init_xy_mode in {
            'corner',
            'sample-uniformly-xy-space',
            'sample-from-goal-space',  # soon to be deprecated
        }
        assert not goal_is_xy or not goal_is_qpos
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        if use_low_gear_ratio:
            model_path = "classic_mujoco/ant_maze_gear30_small_dt3_with_invis.xml"
        MujocoEnv.__init__(self,
                           model_path=get_asset_full_path(model_path),
                           frame_skip=frame_skip,
                           **kwargs)
        if goal_is_xy:
            assert reward_type.startswith('xy')

        if init_qpos is not None:
            self.init_qpos[:len(init_qpos)] = np.array(init_qpos)

        self.action_space = Box(-np.ones(8), np.ones(8), dtype=np.float32)
        self.reward_type = reward_type
        self.count_bonus_coeff = count_bonus_coeff
        self.n_bins = n_bins
        self.bin_counts = np.ones((self.n_bins, self.n_bins))
        self.target_radius = target_radius
        self.norm_order = norm_order
        self.goal_is_xy = goal_is_xy
        self.goal_is_qpos = goal_is_qpos
        self.fixed_goal = fixed_goal
        self.diagnostics_goal = diagnostics_goal
        print(f"[Ant] Diagnostics goal: {self.diagnostics_goal}")
        self.init_xy_mode = init_xy_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        # self._healthy_reward = health_reward
        self._done_penalty = done_penalty
        self._healthy_z_range = healthy_z_range

        self.model_path = model_path
        assert goal_sampling_strategy in {'uniform', 'preset1', 'presampled'}
        self.goal_sampling_strategy = goal_sampling_strategy
        if self.goal_sampling_strategy == 'presampled':
            assert presampled_goal_paths is not None
            if not osp.exists(presampled_goal_paths):
                presampled_goal_paths = get_asset_full_path(
                    presampled_goal_paths
                )
            self.presampled_goals = np.load(presampled_goal_paths)
        else:
            self.presampled_goals = None

        self.ant_low, self.ant_high = np.array(ant_low), np.array(ant_high)
        goal_low, goal_high = np.array(goal_low), np.array(goal_high)
        self.two_frames = two_frames
        self.vel_in_state = vel_in_state
        if self.vel_in_state:
            obs_space_low = np.concatenate((self.ant_low, -np.ones(27)))
            obs_space_high = np.concatenate((self.ant_high, np.ones(27)))
            if goal_is_xy:
                goal_space_low = goal_low
                goal_space_high = goal_high
            else:
                goal_space_low = np.concatenate((goal_low, -np.ones(27)))
                goal_space_high = np.concatenate((goal_high, np.ones(27)))
        else:
            obs_space_low = np.concatenate((self.ant_low, -np.ones(13)))
            obs_space_high = np.concatenate((self.ant_high, np.ones(13)))
            if goal_is_xy:
                goal_space_low = goal_low
                goal_space_high = goal_high
            else:
                goal_space_low = np.concatenate((goal_low, -np.ones(13)))
                goal_space_high = np.concatenate((goal_high, np.ones(13)))

        if self.two_frames:
            self.obs_space = Box(np.concatenate((obs_space_low, obs_space_low)),
                                 np.concatenate((obs_space_high, obs_space_high)),
                                 dtype=np.float32)
            self.goal_space = Box(np.concatenate((goal_space_low, goal_space_low)),
                                  np.concatenate((goal_space_high, goal_space_high)),
                                  dtype=np.float32)
        else:
            self.obs_space = Box(obs_space_low, obs_space_high, dtype=np.float32)
            self.goal_space = Box(goal_space_low, goal_space_high, dtype=np.float32)
        qpos_space = Box(-10*np.ones(15), 10*np.ones(15))

        spaces = [
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ]
        if self.goal_is_xy:
            self.xy_obs_space = Box(goal_low, goal_high, dtype=np.float32)
            spaces += [
                ('xy_observation', self.xy_obs_space),
                ('xy_desired_goal', self.xy_obs_space),
                ('xy_achieved_goal', self.xy_obs_space),
            ]
        if self.goal_is_qpos:
            spaces += [
                ('qpos_desired_goal', qpos_space),
                ('qpos_achieved_goal', qpos_space),
            ]

        self.observation_space = Dict(spaces)

        self._full_state_goal = None
        self._xy_goal = None
        self._qpos_goal = None
        self._prev_obs = None
        self._cur_obs = None

    def discretized_states(self, states, bins=None, low=None, high=None):
        """
        Converts continuous to discrete states.
        
        Params
        - states: A shape (n, 2) batch of continuous observations
        - bins: Number of bins for both x and y coordinates
        - low: Lowest value (inclusive) for continuous x and y
        - high: Highest value (inclusive) for continuous x and y
        """
        if bins is None:
            bins = self.n_bins
        if low is None:
            low = self.obs_space.low[0]
        if high is None:
            high = self.obs_space.high[0]

        bin_size = (high - low) / bins
        shifted_states = states - low
        return np.clip(shifted_states // bin_size, 0, bins - 1).astype(np.int32)

    def step(self, action):
        self._prev_obs = self._cur_obs
        self.do_simulation(np.array(action), self.frame_skip)
        ob = self._get_obs()

        # Update bin counts (visitations)
        disc = self.discretized_states(ob['xy_observation'])    
        self.bin_counts[disc[0], disc[1]] += 1

        reward = self.compute_reward(action, ob)
        info = {}
        if self._xy_goal is not None:
            info['xy-distance'] = self._compute_xy_distances(
                self.numpy_batchify_dict(ob)
            )
        if self.count_bonus_coeff:
            info['count_bonus'] = self._count_bonus
        if self.terminate_when_unhealthy:
            done = not self.is_healthy
            if done:
                reward += self._done_penalty
        else:
            done = False
        self._cur_obs = ob
        if len(self.init_qpos) > 15 and self.viewer is not None:
            qpos = self.sim.data.qpos
            qpos[15:] = self._full_state_goal[:15]
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)
        return ob, reward, done, info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    def _get_obs(self):
        qpos = list(self.sim.data.qpos.flat)[:15]
        flat_obs = qpos
        if self.vel_in_state:
            flat_obs = flat_obs + list(self.sim.data.qvel.flat[:14])

        xy = self.sim.data.get_body_xpos('torso')[:2]
        ob = dict(
            observation=flat_obs,
            desired_goal=self._full_state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._full_state_goal,
            state_achieved_goal=flat_obs,
            xy_observation=xy,
            xy_desired_goal=self._xy_goal,
            xy_achieved_goal=xy,
            qpos_desired_goal=self._qpos_goal,
            qpos_achieved_goal=qpos,
        )

        if self.two_frames:
            if self._prev_obs is None:
                self._prev_obs = ob
            ob = self.merge_frames(self._prev_obs, ob)

        # Make sure a copy of the observation is used to avoid aliasing bugs.
        ob = {k: np.array(v) for k, v in ob.items()}
        return ob

    def merge_frames(self, dict1, dict2):
        dict = {}
        for key in dict1.keys():
            dict[key] = np.concatenate((dict1[key], dict2[key]))
        return dict

    def get_goal(self):
        if self.two_frames:
            return {
                'desired_goal': np.concatenate((self._full_state_goal, self._full_state_goal)),
                'state_desired_goal': np.concatenate((self._full_state_goal, self._full_state_goal)),
                'xy_desired_goal': np.concatenate((self._xy_goal, self._xy_goal)),
                'qpos_desired_goal': np.concatenate((self._qpos_goal,
                                                  self._qpos_goal)),
            }
        else:
            goal_dict = {
                'desired_goal': self._full_state_goal,
                'state_desired_goal': self._full_state_goal,
                'xy_desired_goal': self._xy_goal,
                'qpos_desired_goal': self._qpos_goal,
            }
            copied_goal_dict = {}
            for k, v in goal_dict.items():
                if goal_dict[k] is not None:
                    copied_goal_dict[k] = v.copy()
                else:
                    copied_goal_dict[k] = v
            return copied_goal_dict

    def sample_goals(self, batch_size):
        if self.fixed_goal is not None or self.two_frames:
            raise NotImplementedError()
        state_goals = None
        qpos_goals = None
        if self.goal_sampling_strategy == 'uniform':
            assert self.goal_is_xy
            xy_goals = self._sample_uniform_xy(batch_size)
            state_goals = xy_goals
        elif self.goal_sampling_strategy == 'preset1':
            assert self.goal_is_xy
            xy_goals = PRESET1[
                np.random.randint(PRESET1.shape[0], size=batch_size), :
            ]
        elif self.goal_sampling_strategy == 'presampled':
            idxs = np.random.randint(
                self.presampled_goals.shape[0], size=batch_size,
            )
            state_goals = self.presampled_goals[idxs, :]
            xy_goals = state_goals[:, :2]
            qpos_goals = state_goals[:, :15]
        else:
            raise NotImplementedError(self.goal_sampling_strategy)
        goals_dict = {
            'desired_goal': state_goals.copy(),
            'xy_desired_goal': xy_goals.copy(),
            'state_desired_goal': state_goals.copy(),
        }

        return goals_dict

    def _sample_uniform_xy(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low[:2],
            self.goal_space.high[:2],
            size=(batch_size, 2),
        )
        return goals

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'xy_dense':
            r = - self._compute_xy_distances(obs)
        elif self.reward_type == 'xy_sparse':
            r = - (self._compute_xy_distances(obs) > self.target_radius).astype(np.float32)
        elif self.reward_type == 'dense':
            r = - self._compute_state_distances(obs)
        elif self.reward_type == 'qpos_dense':
            r = - self._compute_qpos_distances(obs)
        elif self.reward_type == 'vectorized_dense':
            r = - self._compute_vectorized_state_distances(obs)
        else:
            raise NotImplementedError("Invalid/no reward type.")

        if self.count_bonus_coeff:
            pos_d = self.discretized_states(obs['xy_achieved_goal']) 
            self._count_bonus = self.count_bonus_coeff * 1 / self.bin_counts[pos_d[:, 0], pos_d[:, 1]]
            r += self._count_bonus

        return r

    def get_grid_vals(self, bins=16, low=-4, high=4):
        xs = np.linspace(low, high, bins)
        ys = np.linspace(low, high, bins)
        xys = np.meshgrid(xs, ys)
        grid_vals = np.array(xys).transpose(1, 2, 0).reshape(-1, 2)
        return grid_vals

    def get_grid_count_bonuses(self):
        grid_vals = self.discretized_states(self.get_grid_vals(bins=self.n_bins, low=self.obs_space.low[0], high=self.obs_space.high[0]))
        count_bonuses = self.count_bonus_coeff * 1 / np.sqrt(self.bin_counts[grid_vals[:, 0], grid_vals[:, 1]])
        return count_bonuses.reshape(self.n_bins, self.n_bins)

    def _compute_xy_distances(self, obs):
        achieved_goals = obs['xy_achieved_goal']
        desired_goals = obs['xy_desired_goal']
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.linalg.norm(diff, ord=self.norm_order, axis=1)

    def _compute_qpos_distances(self, obs):
        achieved_goals = obs['qpos_achieved_goal']
        desired_goals = obs['qpos_desired_goal']
        if desired_goals.shape == (1,):
            return -1000
        return np.linalg.norm(
            achieved_goals - desired_goals, ord=self.norm_order, axis=1
        )

    def _compute_vectorized_state_distances(self, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        ant_pos = achieved_goals
        goals = desired_goals
        diff = ant_pos - goals
        return np.abs(diff)

    def reset_model(self, goal=None):
        if goal is None:
            goal = self.sample_goal()
        self._reset_ant()
        self._set_goal(goal)
        self.sim.forward()
        self._prev_obs = None
        self._cur_obs = None
        return self._get_obs()

    def _reset_ant(self):
        qpos = self.init_qpos
        qvel = np.zeros_like(self.init_qvel)
        if self.init_xy_mode == 'sample-uniformly-xy-space':
            xy_start = self._sample_uniform_xy(1)[0]
            qpos[:2] = xy_start
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        self._set_goal(goal)

    def _set_goal(self, goal):
        if 'state_desired_goal' in goal:
            self._full_state_goal = goal['state_desired_goal']
            self._qpos_goal = self._full_state_goal[:15]
            self._xy_goal = self._qpos_goal[:2]
            if 'qpos_desired_goal' in goal:
                assert (self._qpos_goal == goal['qpos_desired_goal']).all()
            if 'xy_desired_goal' in goal:
                assert (self._xy_goal == goal['xy_desired_goal']).all()
        elif 'qpos_desired_goal' in goal:
            self._full_state_goal = None
            self._qpos_goal = goal['qpos_desired_goal']
            self._xy_goal = self._qpos_goal[:2]
            if 'xy_desired_goal' in goal:
                assert (self._xy_goal == goal['xy_desired_goal']).all()
        elif 'xy_desired_goal' in goal:
            self._full_state_goal = None
            self._qpos_goal = None
            self._xy_goal = goal['xy_desired_goal']
        else:
            raise ValueError("C'mon, you gotta give me some goal!")
        assert self._xy_goal is not None
        self._prev_obs = None
        self._cur_obs = None
        if len(self.init_qpos) > 15 and self._qpos_goal is not None:
            qpos = self.init_qpos
            qpos[15:] = self._qpos_goal
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)

    def get_env_state(self):
        ant_qpos = self.data.qpos.flat.copy()[:15]
        goal = self.get_goal().copy()
        return ant_qpos, goal

    def set_env_state(self, state):
        ant_qpos, goal = state
        self._set_goal(goal)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[:15] = ant_qpos
        qvel[:15] = 0
        self.set_state(qpos, qvel)

    def set_to_goal(self, goal):
        import warnings
        warnings.warn("Ignoring set to goal")
        # qpos = self.data.qpos.flat.copy()
        # qvel = self.data.qvel.flat.copy()
        # qpos[:15] = goal['qpos_desired_goal']
        # qvel[:15] = 0
        # self.set_state(qpos, qvel)


if __name__ == '__main__':
    env = AntEnv(
        model_path='classic_mujoco/ant_maze.xml',
        goal_low=[-1, -1],
        goal_high=[6, 6],
        goal_is_xy=True,
        init_qpos=[
            0, 0, 0.5, 1,
            0, 0, 0,
            0,
            1.,
            0.,
            -1.,
            0.,
            -1.,
            0.,
            1.,
        ],
        reward_type='xy_dense',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        env.step(action)
        if i % 10 == 0:
            env.reset()
