from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color
import collections

from skimage import transform
import skimage

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall

N_SHUFFLE_BINS = 16
MAZE_LOW = -4
MAZE_HIGH = 4

class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=256,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.5,
            boundary_dist=4,
            ball_radius=0.15,
            ball_pixel_radius=0,
            walls=None,
            init_pos_range=None,
            target_pos_range=None,
            sparse_goals=None,
            shuffle_states=False,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            n_bins=10,
            use_count_reward=False,
            show_discrete_grid=False,
            convert_obs_to_image=False,
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        # if fixed_goal is not None:
        #     fixed_goal = np.array(fixed_goal)

        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.ball_pixel_radius = ball_pixel_radius
        self.walls = walls
        self.n_bins = n_bins
        self.use_count_reward = use_count_reward
        self.show_discrete_grid = show_discrete_grid
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
        self.sparse_goals = sparse_goals
        self.shuffle_states = shuffle_states
        self.convert_obs_to_image = convert_obs_to_image

        self.x_bins = np.linspace(-self.boundary_dist, self.boundary_dist, self.n_bins)
        self.y_bins = np.linspace(-self.boundary_dist, self.boundary_dist, self.n_bins)
        self.bin_counts = np.ones((self.n_bins + 1, self.n_bins + 1))

        self._random_mapping_x = np.random.permutation(range(N_SHUFFLE_BINS))
        self._random_mapping_y = np.random.permutation(range(N_SHUFFLE_BINS))

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros((2))

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')

        print(f"[Point2D] Using boundary dist: {self.boundary_dist}, obs space: ({self.obs_range.low}, {self.obs_range.high})")

        if not init_pos_range:
            self.init_pos_range = self.obs_range
        else:
            assert np.all(np.abs(init_pos_range) < boundary_dist), (f"Init position must be"
                "within the boundaries of the environment: ({-boundary_dist}, {boundary_dist})")
            low, high = init_pos_range
            self.init_pos_range = spaces.Box(
                np.array(low), np.array(high), dtype='float32')
            print(f"[Point2D] Using initial pos range of low={low}, high={high}")

        if not target_pos_range:
            self.target_pos_range = self.obs_range
        else:
            assert np.all(np.abs(target_pos_range) < boundary_dist), (f"Goal position must be"
                "within the boundaries of the environment: ({-boundary_dist}, {boundary_dist})")

            low, high = target_pos_range
            self.target_pos_range = spaces.Box(
                np.array(low), np.array(high), dtype='float32')

        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('onehot_observation', spaces.Box(
                0, 1, shape=(2 * (self.n_bins + 1), ), dtype=np.float32)),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        self.drawer = None
        self.render_drawer = None

        self.reset()

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        if sum(new_position != orig_new_pos) > 1:
            """
            Hack: sometimes you get caught on two walls at a time. If you
            process the input in the other direction, you might only get
            caught on one wall instead.
            """
            new_position = orig_new_pos.copy()
            for wall in self.walls[::-1]:
                new_position = wall.handle_collision(
                    self._position, new_position
                )

        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        x_d, y_d = ob['discrete_observation']
        self.bin_counts[x_d, y_d] += 1

        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
            'ext_reward': self._ext_reward,
        }

        if self.sparse_goals is not None:
            # Set reward for anything within threshold of goal(s) to 1
            # (ignore any classifier reward)
            goals, threshold = self.sparse_goals
            min_dist = self.compute_min_dist(goals)
            info['is_sparse_success'] = min_dist < threshold

        if hasattr(self, 'wall_shape'):
            if self.wall_shape == 'medium-maze':
                info['manhattan_dist_to_target'] = self._medium_maze_manhattan_distance(ob['state_achieved_goal'], ob['state_desired_goal'])
            elif self.wall_shape == 'hard-maze':
                info['manhattan_dist_to_target'] = self._hard_maze_goal_distance(ob['state_achieved_goal'], ob['state_desired_goal'])
            elif self.wall_shape == 'double-medium-maze':
                goals, threshold = self.sparse_goals
                info['manhattan_dist_to_sparse_goal'] = self.compute_min_dist(goals, self._double_maze_manhattan_distance)
                info['manhattan_dist_to_target'] = self.compute_min_dist(ob['state_desired_goal'], self._double_maze_manhattan_distance)

        if self.use_count_reward:
            info['count_bonus'] = self._count_bonus

        done = False
        return ob, reward, done, info

    def compute_min_dist(self, goals, 
                         distance_fn=lambda s1, s2: np.linalg.norm(s1 - s2, axis=-1)):
        dists = []
        if len(goals.shape) == 1:
            goals = goals[None]
        for goal in goals:
            dist = distance_fn(self._position, goal)
            dists.append(dist)
        min_dist = np.array(dists).min(axis=0)
        return min_dist

    def reset(self):
        # TODO: Make this more general
        self._target_position = self.sample_goal()['state_desired_goal']
        # if self.randomize_position_on_reset:
        self._position = self._sample_position(
            # self.obs_range.low,
            # self.obs_range.high,
            self.init_pos_range.low,
            self.init_pos_range.high,
        )
        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        if np.all(low == high):
            return low
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def clear_bin_counts(self):
        self.bin_counts = np.ones((self.n_bins + 1, self.n_bins + 1))

    def _discretize_observation(self, obs):
        x, y = obs['state_observation'].copy()
        x_d, y_d = np.digitize(x, self.x_bins), np.digitize(y, self.y_bins)
        return np.array([x_d, y_d])

    def _discretized_states(self, states, bins=16, low=-4, high=4):
        """
        Converts continuous to discrete states.
        
        Params
        - states: A shape (n, 2) batch of continuous observations
        - bins: Number of bins for both x and y coordinates
        - low: Lowest value (inclusive) for continuous x and y
        - high: Highest value (inclusive) for continuous x and y
        """
        bin_size = (high - low) / bins
        shifted_states = states - low
        return np.clip(shifted_states // bin_size, 0, bins - 1).astype(np.int32)

    def _get_shuffled_states(self, states):
        if len(states.shape) == 1:
            states = states[None]
        states = self._discretized_states(states, bins=N_SHUFFLE_BINS, low=MAZE_LOW, high=MAZE_HIGH)
        results = (np.hstack([self._random_mapping_x[states[:,0]][:,None], 
                              self._random_mapping_y[states[:,1]][:,None]]) - 8) / 2
        return results

    def _state_to_pixel_coords(self, states, state_min=-4, state_max=4, img_width=28):
        coords = (states - state_min) * (img_width / (state_max - state_min))
        
        # Need to subtract 1 because pygame rendering shifts everything to the left/up by 1 for some reason
        # (e.g. [-3, -3] is at 5, 5 instead of 6, 6). However this does not apply to the top/left edges.
        return np.clip(coords, 0, None).astype(np.int32)

    def _get_obs(self):
        state_obs = self._get_shuffled_states(self._position.copy()).squeeze() \
            if self.shuffle_states else self._position.copy()
            
        obs = collections.OrderedDict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=state_obs,
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._position.copy(),
        )

        # Update with discretized state
        pos_discrete = self._discretize_observation(obs)
        pos_onehot = np.eye(self.n_bins + 1)[pos_discrete]
        obs['discrete_observation'] = pos_discrete
        obs['onehot_observation'] = pos_onehot

        return obs

    def _hard_maze_goal_distance(self, s1, goal=None):
        s1 = s1.copy()
        if len(s1.shape) == 1:
            s1 = s1[None]

        x1,  y1 = s1[:,0], s1[:,1]
        dist = np.zeros(len(s1))
        
        top_section = y1 <= -2
        dist[top_section] += np.abs(2 - x1[top_section]) # Move horizontally to corner
        dist[top_section] += -2 - y1[top_section] # Move vertically to corner
        x1[top_section], y1[top_section] = 2, -2
        
        right_section = x1 >= 2
        dist[right_section] += x1[right_section] - 2 # Move horizontally
        dist[right_section] += np.abs(2 - y1[right_section]) # Move vertically to corner
        x1[right_section], y1[right_section] = 2, 2
        
        bottom_section = y1 >= 2
        dist[bottom_section] += np.abs(-2 - x1[bottom_section]) # Move horizontally
        dist[bottom_section] += y1[bottom_section] - 2 # Move vertically to corner
        x1[bottom_section], y1[bottom_section] = -2, 2
        
        left_section = x1 <= -2
        dist[left_section] += (-2 - x1[left_section]) # Move horizontally
        dist[left_section] += np.abs(y1[left_section]) # Move vertically to corner
        x1[left_section], y1[left_section] = -2, 0
        
        mid_section = np.logical_and(y1 <= 0, y1 >= -2)
        dist[mid_section] += np.abs(x1[mid_section]) # Move horizontally
        dist[mid_section] += np.abs(y1[mid_section]) # Move vertically to corner
        
        # Move to goal!!!
        dist += np.abs(x1 - (-0.5))
        dist += np.abs(y1 - 1.25)
        
        return dist

    def _medium_maze_manhattan_distance(self, s1, s2):
        # Maze wall positions
        left_wall_x = -self.boundary_dist/3
        left_wall_bottom = self.inner_wall_max_dist
        right_wall_x = self.boundary_dist/3
        right_wall_top = -self.inner_wall_max_dist
        
        s1 = s1.copy()
        s2 = s2.copy()

        if len(s1.shape) == 1:
            s1, s2 = s1[None], s2[None]
        
        # Since maze distances are symmetric, redefine s1,s2 for convenience 
        # so that points in s1 are to the left of those in s2
        combined = np.hstack((s1[:,None], s2[:,None]))
        indices = np.argmin((s1[:,0], s2[:,0]), axis=0)
        s1 = np.take_along_axis(combined, indices[:,None,None], axis=1).squeeze(axis=1)
        s2 = np.take_along_axis(combined, 1 - indices[:,None,None], axis=1).squeeze(axis=1)
        
        x1 = s1[:,0]
        x2 = s2[:,0]
        
        # Horizontal movement
        x_dist = np.abs(x2 - x1)
        
        # Vertical movement
        boundary_ys = [left_wall_bottom, right_wall_top, 0]
        boundary_xs = [left_wall_x, right_wall_x, self.boundary_dist, -4.0001]
        y_directions = [1, -1, 0] # +1 means need to get to bottom, -1 means need to get to top
        curr_y, goal_y = s1[:,1], s2[:,1]
        y_dist = np.zeros(len(s1))
        
        for i in range(3):
            # Get all points where s1 and s2 respectively are in the current vertical section of the maze
            curr_in_section = x1 <= boundary_xs[i]
            goal_in_section = np.logical_and(boundary_xs[i-1] < x2, x2 <= boundary_xs[i])
            goal_after_section = x2 > boundary_xs[i]
            
            # Both in current section: move directly to goal
            mask = np.logical_and(curr_in_section, goal_in_section)
            y_dist += mask * np.abs(curr_y - goal_y)
            curr_y[mask] = goal_y[mask]
            
            # s2 is further in maze: move to next corner
            mask = np.logical_and(curr_in_section, np.logical_and(goal_after_section, y_directions[i] * (boundary_ys[i] - curr_y) > 0))
            y_dist += mask * np.clip(y_directions[i] * (boundary_ys[i] - curr_y), 0, None)
            curr_y[mask] = boundary_ys[i]
            
        return x_dist + y_dist

    def _double_maze_manhattan_distance(self, s1, s2):
        # Maze wall positions
        left_wall_bottom = self.inner_wall_max_dist + 1
        right_wall_top = -self.inner_wall_max_dist - 1

        s1 = s1.copy()
        s2 = s2.copy()

        if len(s1.shape) == 1:
            s1 = s1[None]
        if len(s2.shape) == 1:
            s2 = s2[None]

        # Since maze distances are symmetric, redefine s1,s2 for convenience 
        # so that points in s1 are to the left of those in s2
        combined = np.hstack((s1[:,None], s2[:,None]))
        indices = np.argmin((s1[:,0], s2[:,0]), axis=0)
        s1 = np.take_along_axis(combined, indices[:,None,None], axis=1).squeeze(axis=1)
        s2 = np.take_along_axis(combined, 1 - indices[:,None,None], axis=1).squeeze(axis=1)

        x1 = s1[:,0]
        x2 = s2[:,0]

        # Horizontal movement
        x_dist = np.abs(x2 - x1)

        # Vertical movement
        boundary_ys = [right_wall_top, left_wall_bottom, right_wall_top, left_wall_bottom, 0]
        boundary_xs = [-3/5 * self.boundary_dist, -1/5 * self.boundary_dist, 1/5 * self.boundary_dist, 3/5 * self.boundary_dist, self.boundary_dist, -4.0001]
        y_directions = [-1, +1, -1, +1, 0] # +1 means need to get to bottom, -1 means need to get to top
        curr_y, goal_y = s1[:,1], s2[:,1]
        y_dist = np.zeros(len(s1))

        for i in range(5):
            # Get all points where s1 and s2 respectively are in the current vertical section of the maze
            curr_in_section = x1 <= boundary_xs[i]
    #         print(x1)
    #         print(boundary_xs[i])
            goal_in_section = np.logical_and(boundary_xs[i-1] < x2, x2 <= boundary_xs[i])
            goal_after_section = x2 > boundary_xs[i]
            
    #         print("x1:", x1)
    #         print("x2:", x2)
    #         print(boundary_xs[i])
    #         print(boundary_xs[i-1])
    #         print(curr_in_section, goal_in_section)

            # Both in current section: move directly to goal
            mask = np.logical_and(curr_in_section, goal_in_section)
            y_dist += mask * np.abs(curr_y - goal_y)
            curr_y[mask] = goal_y[mask]

            # s2 is further in maze: move to next corner
            mask = np.logical_and(curr_in_section, np.logical_and(goal_after_section, y_directions[i] * (boundary_ys[i] - curr_y) > 0))
            y_dist += mask * np.clip(y_directions[i] * (boundary_ys[i] - curr_y), 0, None)
            curr_y[mask] = boundary_ys[i]

    #     print(x_dist, y_dist)
        return x_dist + y_dist

    def compute_rewards(self, actions, obs, reward_type=None):
        reward_type = reward_type or self.reward_type

        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if reward_type == "sparse":
            r = -(d > self.target_radius).astype(np.float32)
        elif reward_type == "multi-sparse":
            goals, threshold = self.sparse_goals
            min_dist = np.array([self.compute_min_dist(goals)])
            r = -(min_dist > self.target_radius).astype(np.float32)
        elif reward_type == "dense":
            r = -d
        elif reward_type == "vectorized_dense":
            r = -np.abs(achieved_goals - desired_goals)
        elif reward_type == "medium_maze_ground_truth_manhattan":
            """
            # Use maze distance from current position to goal as the negative reward.
            # This is specifically for the medium Maze-v0 env, which has two vertical walls.
            """
            r = -self._medium_maze_manhattan_distance(achieved_goals, desired_goals)
        elif reward_type == "laplace_smoothing":
            # Laplace smoothing: 1 within the goal region, 1/(n+2) at all other states
            # (where n is the number of visitations)
            r = np.zeros(d.shape)
            pos_d = obs['discrete_observation']
            r = 1 / (self.bin_counts[pos_d[:,0], pos_d[:,1]] + 2)
            r[d <= self.target_radius] = 1
        elif reward_type == "none":
            r = np.zeros(d.shape)
        else:
            raise NotImplementedError(f"Unrecognized reward type: {reward_type}")

        if self.use_count_reward:
            # TODO: Add different count based strategies
            pos_d = obs['discrete_observation']
            self._ext_reward = r.copy()
            self._count_bonus = 1 / np.sqrt(self.bin_counts[pos_d[:, 0], pos_d[:, 1]])
            r += self._count_bonus
        else:
            self._ext_reward = r.copy()

        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'radius',
            'target_position',
            'distance_to_target',
            'velocity',
            'speed',
            'is_success',
            'ext_reward',
            'count_bonus',
        ] + (['is_sparse_success'] if self.sparse_goals is not None else []):
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def sample_goals(self, batch_size):
        # if self.fixed_goal:
        #     goals = np.repeat(
        #         self.fixed_goal.copy()[None],
        #         batch_size,
        #         0)
        # else:
        goals = np.zeros((batch_size, self.obs_range.low.size))
        for b in range(batch_size):
            if batch_size > 1:
                logging.warning("This is very slow!")
            goals[b, :] = self._sample_position(
                # self.obs_range.low,
                # self.obs_range.high,
                self.target_pos_range.low,
                self.target_pos_range.high,
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None, invert_colors=False):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                # TODO: Action scale = 1 breaks rendering, why?
                # x_bounds=(-self.boundary_dist - self.ball_radius,
                #           self.boundary_dist + self.ball_radius),
                # y_bounds=(-self.boundary_dist - self.ball_radius,
                #           self.boundary_dist + self.ball_radius),
                x_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                y_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                render_onscreen=self.render_onscreen,
            )

        old_position = self._position
        self._position = np.array([-4, -4])

        self.draw(self.drawer)
        img = self.drawer.get_image()
        # img = skimage.transform.resize(img, (width, height), anti_aliasing=True, preserve_range=True)
        self._position = old_position

        if self.images_are_rgb:
            im = img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            im = (-r + b).transpose().flatten()

        im[0:2, 0:2] = np.array([255, 255, 255])
        im = 255 - im # Invert image - make background black, walls white
        im = im[None]
        
        # Set pixel at ball position to blue for each individual observation
        pixel_coords = self._state_to_pixel_coords(self._position, img_width=width)[None]
        for i in range(-self.ball_pixel_radius, self.ball_pixel_radius+1):
            for j in range(-self.ball_pixel_radius, self.ball_pixel_radius+1):    
                im[range(len(pixel_coords)), np.clip(pixel_coords[:,1]+i, 0, 27), np.clip(pixel_coords[:,0]+j, 0, 27)] = np.array([0, 0, 255])
                
        self._position = old_position
        return im[0] / 255

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._position = goal
        self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._position = position
        self._target_position = goal

    def draw(self, drawer):
        drawer.fill(Color('white'))

        if self.show_discrete_grid:
            for x in self.x_bins:
                drawer.draw_segment(
                    (x, -self.boundary_dist),
                    (x, self.boundary_dist),
                    Color(220,220,220,25), aa=False)
            for y in self.y_bins:
                drawer.draw_segment(
                    (-self.boundary_dist, y),
                    (self.boundary_dist, y),
                    Color(220,220,220,25), aa=False)

        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        try:
            drawer.draw_solid_circle(
                self._position,
                self.ball_radius,
                Color('blue'),
            )
        except ValueError as e:
            print('\n\n RENDER ERROR \n\n')

        for wall in self.walls:
            drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )
        drawer.render()

    def render(self, mode='human', width=None, height=None, invert_colors=False, close=False):
        if close:
            self.render_drawer = None
            return
        if mode =='rgb_array':
            return self.get_image(width=width, height=height, invert_colors=invert_colors)

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                # x_bounds=(-self.boundary_dist-self.ball_radius,
                #           self.boundary_dist+self.ball_radius),
                # y_bounds=(-self.boundary_dist-self.ball_radius,
                #           self.boundary_dist+self.ball_radius),
                x_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                y_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                render_onscreen=True,
            )
        self.draw(self.render_drawer)
        self.render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self.render_drawer.check_for_exit()

    # def get_diagnostics(self, paths, prefix=''):
    #     statistics = OrderedDict()
    #     for stat_name in ('distance_to_target', ):
    #         stat_name = stat_name
    #         stat = get_stat_in_paths(paths, 'env_infos', stat_name)
    #         statistics.update(create_stats_ordered_dict(
    #             '%s%s' % (prefix, stat_name),
    #             stat,
    #             always_show_all_stats=True,
    #             ))
    #         statistics.update(create_stats_ordered_dict(
    #             'Final %s%s' % (prefix, stat_name),
    #             [s[-1] for s in stat],
    #             always_show_all_stats=True,
    #             ))
    #     return statistics

    """Static visualization/utility methods"""

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )
        ax.set_xlim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )

    def initialize_camera(self, init_fctn):
        pass

class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            wall_shape="hard-maze",
            wall_thickness=1.0,
            inner_wall_max_dist=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness

        WALL_FORMATIONS = {
            "u": [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ],
            "-": [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ],
            "--": [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ],
            "big-u": [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ],
            "easy-u": [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ],
            "big-h": [
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                ),
            ],
            "box": [
                # Bottom wall
                VerticalWall(
                    self.ball_radius,
                    0,
                    0,
                    0,
                    self.wall_thickness
                ),
            ],
            "easy-maze": [
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
            ],
            "medium-maze": [
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist/3,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    self.boundary_dist/3,
                    -self.inner_wall_max_dist,
                    self.boundary_dist
                ),
            ],
            "double-medium-maze": [
                VerticalWall(
                    self.ball_radius,
                    -3/5 * self.boundary_dist,
                    -self.inner_wall_max_dist - 1,
                    self.boundary_dist
                ),
                VerticalWall(
                    self.ball_radius,
                    -1/5 * self.boundary_dist,
                    -self.boundary_dist,
                    self.inner_wall_max_dist + 1,
                ),
                VerticalWall(
                    self.ball_radius,
                    1/5 * self.boundary_dist,
                    -self.inner_wall_max_dist - 1,
                    self.boundary_dist
                ),
                VerticalWall(
                    self.ball_radius,
                    3/5 * self.boundary_dist,
                    -self.boundary_dist,
                    self.inner_wall_max_dist + 1,
                ),
            ],
            "hard-maze": [
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    self.boundary_dist - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist - self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist * 2,
                    self.boundary_dist - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist * 2,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    0,
                ),
            ],
            "horizontal-maze": [
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist/2,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.boundary_dist
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist/2,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
            ],
            None: [],
        }

        self.walls = WALL_FORMATIONS.get(wall_shape, [])

if __name__ == "__main__":
    import gym
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import multiworld
    multiworld.register_all_envs()

    e = gym.make('Point2DFixed-v0', **{'reward_type': 'none', 'use_count_reward': True})
    # e = gym.make('Point2DSingleWall-v0')

    # e = gym.make('Point2D-Box-Wall-v1')
    # e = gym.make('Point2D-Big-UWall-v1')
    # e = gym.make('Point2D-Easy-UWall-v1')
    # e = gym.make('Point2DEnv-Image-v0')

    for i in range(100):
        e.reset()
        for j in range(100):
            obs, rew, done, info = e.step(e.action_space.sample())
            # e.render()
            # img = e.get_image()
            # plt.imshow(img)
            # plt.show()
            # print(rew)
            print(e.observation_space, obs['onehot_observation'])
