import numpy as np

from softlearning.environments.utils import get_environment_from_params
import pickle
import os
import matplotlib.pyplot as plt

from softlearning.misc.utils import PROJECT_PATH
goal_directory = os.path.join(PROJECT_PATH, 'goal_classifier')

PICK_TASKS = [
    'PickAndPlace3DEnv-v0',
    'StateSawyerPickAndPlaceEnv-v0',
    'Image48SawyerPickAndPlaceEnv-v0',
    'StateSawyerPickAndPlace3DEnv-v0',
    'Image48SawyerPickAndPlace3DEnv-v0',
]

ANT_MAZE_TASKS = [
    'MazeSEnv-v0',
    'MazeOneWallEnv-v0',
    'v0',
]

DOOR_TASKS = [
    'DoorPullHookEnv-v0',
    'StateSawyerDoorPullHookEnv-v0',
    'Image48SawyerDoorPullHookEnv-v0',
]

PUSH_TASKS = [
    'PushForwardEnv-v0',
    'StateSawyerPushSidewaysEnv-v0',
    'Image48SawyerPushSidewaysEnv-v0',
    'StateSawyerPushForwardEnv-v0',
    'Image48SawyerPushForwardEnv-v0',
]

GOAL_PATH_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Point2D': {
            'Fixed-v0': 'pointmass_nowalls/center',
            'SingleWall-v0': 'pointmass_nowalls/bottom_right',
            'BoxWall-v1': 'pointmass_nowalls/bottom_middle',
        },
        'DClaw': {
            'TurnFreeValve3ResetFree-v0': 'free_screw_180',
            'TurnFreeValve3Fixed-v0': 'free_screw_180',
            'TurnFixed-v0': 'fixed_screw_180',
            'TurnFreeValve3Hardware-v0': 'fixed_screw_180',
            'TurnMultiGoalResetFree-v0': 'fixed_screw_2_goals_mixed_pool_goal_index',
            'LiftDDResetFree-v0': 'dodecahedron_lifting_flat_bowl_arena_red',
            'SlideBeadsResetFree-v0': '4_beads_475',
        },
        'SawyerDhandInHand': {
            'Valve3RepositionFixed-v0': 'valve_reposition',
            'Valve3PickupFixed-v0': 'valve_pickup',
        }
    },
}


def get_ddl_goal_state_from_variant(variant):
    train_env_params = variant['environment_params']['training']
    env = get_environment_from_params(train_env_params)

    universe = train_env_params['universe']
    domain = train_env_params['domain']
    task = train_env_params['task']

    if task in ['Valve3PickupFixed-v0']:
        try:
            env_path = os.path.join(
                goal_directory,
                GOAL_PATH_PER_UNIVERSE_DOMAIN_TASK[universe][domain][task])
            pkl_path = os.path.join(env_path, 'positives.pkl')
            with open(pkl_path, 'rb') as f:
                goal_state = pickle.load(f)
        except KeyError:
            raise NotImplementedError
    else:
        domain_generators = SUPPORTED_ENVS_UNIVERSE_DOMAIN_TASK[universe][domain]
        gen_func = domain_generators.get(task, domain_generators[DEFAULT_TASK_KEY])

        goal_state = gen_func(env,
                            include_transitions=False,
                            num_total_examples=1,
                            goal_threshold=0.0)
    goal_state = {
        key: val[0]
        for key, val in goal_state.items()
    }
    return goal_state


def get_goal_transitions_from_variant(variant):
    """
    Returns SQIL goal transitions (s, a, s', r = 1)
    """
    train_env_params = variant['environment_params']['training']

    env = get_environment_from_params(train_env_params)

    universe = train_env_params['universe']
    domain = train_env_params['domain']
    task = train_env_params['task']

    try:
        # TODO: Add goal generation kwargs (goal threshold, etc.)
        domain_generators = SUPPORTED_ENVS_UNIVERSE_DOMAIN_TASK[universe][domain]
        gen_func = domain_generators.get(task, domain_generators[DEFAULT_TASK_KEY])
        goal_transitions = gen_func(env, include_transitions=True)
    except KeyError:
        raise NotImplementedError

    return goal_transitions

def generate_pusher_2d_goals(env,
                             num_total_examples=500,
                             rollout_length=15,
                             goal_threshold=0.15,
                             include_transitions=True,
                             save_image=True):
    # TODO: Make goal collecting better for both VICE and SQIL
    from copy import deepcopy
    from gym.spaces import Box
    env = deepcopy(env)
    # === Modify the init range ===
    env.unwrapped.set_init_qpos_range(Box(
        np.concatenate([
            env.target_pos_range.low - 0.15,
            np.array([-np.pi])]),
        np.concatenate([
            env.target_pos_range.high + 0.15,
            np.array([np.pi])]),
        dtype=np.float32))
    env.unwrapped.set_init_object_pos_range(Box(env.target_pos_range.low - goal_threshold,
                                                env.target_pos_range.high + goal_threshold,
                                                dtype='float32'))

    if goal_threshold == 0.0 and not include_transitions:
        # Collect a single goal for DDL (align the gripper correctly)
        low = env.target_pos_range.low
        high = env.target_pos_range.high
        rand_target_xy = np.random.uniform(low - goal_threshold, high + goal_threshold)
        rand_gripper_xy = np.random.uniform(low - 0.15, high + 0.15)
        diff = rand_target_xy - rand_gripper_xy
        angle = np.arctan2(diff[1], diff[0])

        env.unwrapped.set_init_qpos_range(
            Box(np.append(rand_gripper_xy, angle),
                np.append(rand_gripper_xy, angle),
                dtype=np.float32))
        env.unwrapped.set_init_object_pos_range(
            Box(rand_target_xy,
                rand_target_xy,
                dtype=np.float32))

        obs = env.reset()
        plt.figure(figsize=(4, 4))
        plt.imshow(env.render(mode='rgb_array', width=256, height=256))
        path = os.path.join(os.getcwd(), 'env_frame.jpg')
        plt.savefig(path)
        return {
            k: v[None]
            for k, v in obs.items()
        }

    observations = []
    actions = []
    next_observations = []

    num_positives = 0
    while num_positives <= num_total_examples:
        # === Initialize variables ===
        prev_obs = env.reset()
        last_action = env.action_space.sample()
        obs, rew, done, info = env.step(last_action)
        t = 0
        while t < rollout_length:
            action = env.action_space.sample()
            prev_obs = obs
            # === GOAL CRITERIA ===
            if env.unwrapped.compute_reward(action, env.unwrapped._get_obs()) < goal_threshold:
                observations.append(prev_obs)
                obs, rew, done, info = env.step(action)
                next_observations.append(obs)
                actions.append(action)
                num_positives += 1
            else:
                obs, rew, done, info = env.step(action)
            t += 1

    # === Package goals in dicts ===
    goal_obs = {
        key: np.concatenate([
            obs[key][None] for obs in observations
        ], axis=0)
        for key in observations[0].keys()
    }
    goal_next_obs = {
        key: np.concatenate([
            obs[key][None] for obs in next_observations
        ], axis=0)
        for key in next_observations[0].keys()
    }
    goal_actions = np.vstack(actions)

    if save_image:
        plt.figure(figsize=(4, 4))
        plt.imshow(env.render(mode='rgb_array'))
        path = os.path.join(os.getcwd(), 'env_frame.jpg')
        plt.savefig(path)

    if include_transitions:
        goal_transitions = {
            'observations': goal_obs,
            'next_observations': goal_next_obs,
            'actions': goal_actions,
        }
        return goal_transitions
    else:
        return goal_obs


def generate_point_2d_goals(env,
                            num_total_examples=500,
                            rollout_length=15,
                            goal_threshold=0.05,
                            include_transitions=True,
                            save_image=True):
    from copy import deepcopy
    from gym.spaces import Box
    env = deepcopy(env)
    # === Modify the init range ===
    env.unwrapped.init_pos_range = Box(env.target_pos_range.low - goal_threshold,
                                       env.target_pos_range.high + goal_threshold,
                                       dtype='float32')

    if goal_threshold == 0.0 and not include_transitions:
        obs = env.reset()
        return {
            k: v[None]
            for k, v in obs.items()
        }

    observations = []
    actions = []
    next_observations = []

    num_positives = 0
    while num_positives <= num_total_examples:
        # === Initialize variables ===
        prev_obs = env.reset()
        last_action = env.action_space.sample()
        obs, rew, done, info = env.step(last_action)
        t = 0
        while t < rollout_length:
            action = env.action_space.sample()
            prev_obs = obs
            # === GOAL CRITERIA ===
            if info['distance_to_target'] < goal_threshold:
                observations.append(prev_obs)
                save_goal_example(env, num_positives, w=28)
                obs, rew, done, info = env.step(action)
                next_observations.append(obs)
                actions.append(action)
                num_positives += 1
            else:
                obs, rew, done, info = env.step(action)
            t += 1

    # === Package goals in dicts ===
    goal_obs = {
        key: np.concatenate([
            obs[key][None] for obs in observations
        ], axis=0)
        for key in observations[0].keys()
    }
    goal_next_obs = {
        key: np.concatenate([
            obs[key][None] for obs in next_observations
        ], axis=0)
        for key in next_observations[0].keys()
    }
    goal_actions = np.vstack(actions)

    if 'state_observation' in goal_obs and save_image:
        plt.figure(figsize=(4, 4))
        plt.xlim(-env.unwrapped.boundary_dist, env.unwrapped.boundary_dist)
        plt.ylim(-env.unwrapped.boundary_dist, env.unwrapped.boundary_dist)
        plt.gca().invert_yaxis()
        goal_pos = goal_obs['state_observation']
        plt.title(f'Collected goals for {env.unwrapped.__class__.__name__}')
        plt.scatter(goal_pos[:, 0], goal_pos[:, 1], s=2)
        path = os.path.join(os.getcwd(), 'goals.jpg')
        plt.savefig(path)

        plt.figure(figsize=(4, 4))
        plt.imshow(env.render(mode='rgb_array'))
        path = os.path.join(os.getcwd(), 'env_frame.jpg')
        plt.savefig(path)

    if include_transitions:
        goal_transitions = {
            'observations': goal_obs,
            'next_observations': goal_next_obs,
            'actions': goal_actions,
        }
        return goal_transitions
    else:
        return goal_obs


def get_goal_example_from_variant(variant):
    train_env_params = variant['environment_params']['training']

    env = get_environment_from_params(train_env_params)
    total_goal_examples = (variant['data_params']['n_goal_examples']
                           + variant['data_params']['n_goal_examples_validation_max'])

    universe = train_env_params['universe']
    domain = train_env_params['domain']
    task = train_env_params['task']

    if task in ['TurnFixed-v0','TurnFreeValve3Fixed-v0', 'Valve3RepositionFixed-v0', 'Valve3PickupFixed-v0']:
        # from IPython import embed; embed()
        try:
            env_path = os.path.join(
                goal_directory,
                GOAL_PATH_PER_UNIVERSE_DOMAIN_TASK[universe][domain][task])
            pkl_path = os.path.join(env_path, 'positives.pkl')
            with open(pkl_path, 'rb') as f:
                goal_examples = pickle.load(f)
        except KeyError:
            raise NotImplementedError
    elif task in DOOR_TASKS:
        goal_examples = generate_door_goal_examples(total_goal_examples, env)
    elif task in PUSH_TASKS:
        goal_examples = generate_push_goal_examples(total_goal_examples, env)
    elif task in PICK_TASKS:
        goal_examples = generate_pick_goal_examples(total_goal_examples, env, 'PickAndPlace3DEnv-v0')
    elif task in ANT_MAZE_TASKS:
        goal_examples = generate_ant_maze_examples(total_goal_examples, env)
    elif SUPPORTED_ENVS_UNIVERSE_DOMAIN_TASK.get(universe, {}).get(domain, None):
        domain_generators = SUPPORTED_ENVS_UNIVERSE_DOMAIN_TASK[universe][domain]
        gen_func = domain_generators.get(task, domain_generators[DEFAULT_TASK_KEY])
        include_transitions = (
            variant['algorithm_params']['type'] == 'VICEDynamicsAware')
        # from IPython import embed; embed()
        goal_examples = gen_func(env,
                         include_transitions=include_transitions,
                         num_total_examples=total_goal_examples)
    else:
        try:
            env_path = os.path.join(
                goal_directory,
                GOAL_PATH_PER_UNIVERSE_DOMAIN_TASK[universe][domain][task])
            pkl_path = os.path.join(env_path, 'positives.pkl')
            with open(pkl_path, 'rb') as f:
                goal_examples = pickle.load(f)
        except KeyError:
            raise NotImplementedError

    n_goal_examples = variant['data_params']['n_goal_examples']
    # total_samples = len(goal_examples[next(iter(goal_examples))])

    # Shuffle the goal images before assigning training/validation
    shuffle = np.random.permutation(total_goal_examples)
    train_indices = shuffle[:n_goal_examples]
    valid_indices = shuffle[n_goal_examples:]

    goal_examples_train = dict([
        (key, {obs_key: value[obs_key][train_indices] for obs_key in value})
        if isinstance(value, dict)
        else (key, value[train_indices])
        for key, value in goal_examples.items()
    ])
    goal_examples_validation = dict([
        (key, {obs_key: value[obs_key][valid_indices] for obs_key in value})
        if isinstance(value, dict)
        else (key, value[valid_indices])
        for key, value in goal_examples.items()
    ])

    return goal_examples_train, goal_examples_validation


def generate_pick_goal_examples(total_goal_examples, env, task_name):
    max_attempt = 50
    top_level_attempts = 10*total_goal_examples
    attempts = 0
    n = 0

    goal_examples = []
    gain = 5.0
    for _ in range(top_level_attempts):
        env.reset()

        for i in range(100):

            if '3D' in task_name:
                obj_xy = env.unwrapped.get_obj_pos()[:2]
                hand_xy = env.unwrapped.get_endeff_pos()[:2]
                goal_xy = env.unwrapped.fixed_goal[3:5]

                hand_obj_distance = np.linalg.norm(obj_xy - 0.02 - hand_xy)
                goal_obj_distance = np.linalg.norm(obj_xy - goal_xy)

                if i < 25:
                    if hand_obj_distance > 0.015:
                        action_xy = gain*(obj_xy - hand_xy)
                    else:
                        action_xy = [0., 0.]
                    action = np.asarray([action_xy[0], action_xy[1], 0., -1])
                elif i < 35:
                    action = np.asarray([0., 0, -1, -1.])
                elif i < 45:
                    action = np.asarray([0., 0, -1,  1.])
                elif i < 60:
                    action = np.asarray([0., 0, +1,  1.])
                elif i < 100:
                    if goal_obj_distance > 0.015:
                        action_xy = gain*(goal_xy - obj_xy)
                    else:
                        action_xy = [0., 0.]
                    action = np.asarray([action_xy[0], action_xy[1], 0., 1.])

            else:

                obj_y = env.unwrapped.get_obj_pos()[1] - 0.02
                hand_y = env.unwrapped.get_endeff_pos()[1]
                goal_y = env.unwrapped.fixed_goal[4]

                if i < 25:
                    if obj_y < (hand_y - 0.01):
                        action = np.asarray([-1., 0., -1.])
                    elif obj_y > (hand_y + 0.01):
                        action = np.asarray([1., 0., -1.])
                    else:
                        action = np.asarray([0., 0., -1.])
                elif i < 40:
                    action = np.asarray([0., -1.0, -1.])
                elif i < 60:
                    action = np.asarray([0., -1.0, 1.0])
                elif i < 80:
                    action = np.asarray([0., 1., 1.])
                elif i < 100:
                    if goal_y < (hand_y - 0.01):
                        action = np.asarray([-1., 0., 1.])
                    elif goal_y > (hand_y + 0.01):
                        action = np.asarray([1., 0., 1.])
                    else:
                        action = np.asarray([0., 0., 1.])

            ob, r, d, info = env.step(action)

        if info['obj_success']:
            n += 1
            goal_examples.append(ob)
            save_goal_example(env, n)

        if len(goal_examples) >= total_goal_examples:
            break

    assert len(goal_examples) == total_goal_examples, (
        f'Could not generate enough goal examples: {len(goal_examples)}')
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples

def generate_ant_maze_examples(total_goal_examples, env, x_noise_min=-1, x_noise_max=0.5, 
                                y_noise_min=-2, y_noise_max=0.5):
    GOAL_QPOS = [5.5, 4.5, 0.565, 1., 0., 0., 0., 0.,
                 1., 0., -1., 0., -1., 0., 1.]
    goal_examples = []
    for i in range(total_goal_examples):
        goal = GOAL_QPOS.copy()
        goal[0] += np.random.uniform(x_noise_min, x_noise_max)
        goal[1] += np.random.uniform(y_noise_min, y_noise_max)
        env.set_env_state((goal, {'xy_desired_goal': goal[:2]}))

        for _ in range(10):
            ob, _, _, _ = env.step(np.random.uniform(-1, 1, size=8))

        save_goal_example(env, i)
        goal_examples.append(ob)

    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    with open('ant_goals.pkl', 'wb') as f:
        pickle.dump(goal_examples, f)

    return goal_examples

def save_goal_example(env, n, w=480):
    # Visualize and save goal examples
    goal_img = env.render(width=w, height=w, mode='rgb_array')
    plt.figure()
    plt.imshow(goal_img)
    if not os.path.exists('./goal_examples'):
        os.mkdir('./goal_examples')
    plt.savefig(f'./goal_examples/goal_{n}.png')

def generate_push_goal_examples(total_goal_examples, env):
    max_attempt = 5*total_goal_examples
    attempts = 0
    n = 0
    goal_examples = []

    while n < total_goal_examples and attempts < max_attempt:

        attempts += 1
        env.reset()
        goal_vec = {
            'state_desired_goal': env.unwrapped.fixed_goal
        }

        goal_vec['state_desired_goal'][:2] += np.random.uniform(low=-0.01, high=0.01, size=(2,))
        goal_vec['state_desired_goal'][-2:] += np.random.uniform(low=-0.01, high=0.01, size=(2,))

        env.unwrapped.set_to_goal(goal_vec)

        endeff_pos = env.unwrapped.get_endeff_pos()
        puck_pos = env.unwrapped.get_puck_pos()

        endeff_distance = np.linalg.norm(endeff_pos - goal_vec['state_desired_goal'][:3])
        puck_distance = np.linalg.norm(puck_pos[:2] - goal_vec['state_desired_goal'][3:5])
        puck_endeff_distance = np.linalg.norm(puck_pos[:2] - endeff_pos[:2])

        endeff_threshold = 0.05
        puck_threshold = env.unwrapped.indicator_threshold
        puck_radius = env.unwrapped.puck_radius

        if (endeff_distance < endeff_threshold
            and puck_distance < puck_threshold
            and puck_endeff_distance > puck_radius):
            ob, rew, done, info = env.step(np.asarray([0., 0.]))
            goal_examples.append(ob)
            n += 1

            save_goal_example(env, n)

    assert len(goal_examples) == total_goal_examples, 'Could not generate enough goal examples'
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples

def generate_door_goal_examples(total_goal_examples, env):

    max_attempt = 10 * total_goal_examples
    attempts = 0
    n = 0
    goal_examples = []

    while n < total_goal_examples and attempts < max_attempt:

        attempts += 1
        env.reset()
        env.unwrapped._set_door_pos(0 + np.random.uniform(low=0., high=0.1))
        goal_vec = {
            'state_desired_goal': env.unwrapped.fixed_goal
        }

        for j in range(100):

            door_angle = env.unwrapped.get_door_angle()
            if j < 25:
                act = [0.05, 1, -0.5]
            elif j < 100 and door_angle < 0.8:
                act = [0.0, -0.4, 0.0]
            else:
                act = [0., 0., 0.]

            act += np.random.uniform(low=-0.01, high=0.01, size=3)
            ob, rew, done, info = env.step(np.asarray(act))

        # goal_vec['state_desired_goal'][:3] += np.random.uniform(low=-0.01, high=0.01, size=(3,))
        # goal_vec['state_desired_goal'][3] += np.random.uniform(low=-0.01, high=0.01)

        # env.unwrapped.set_to_goal_pos(goal_vec['state_desired_goal'][:3])
        # env.unwrapped.set_to_goal_angle(goal_vec['state_desired_goal'][3])

        pos = env.unwrapped.get_endeff_pos()
        angle = env.unwrapped.get_door_angle()
        endeff_distance = np.linalg.norm(pos - goal_vec['state_desired_goal'][:3])
        angle_distance = np.abs(angle - goal_vec['state_desired_goal'][3])
        #state = np.concatenate([pos, angle])
        angle_threshold = env.unwrapped.indicator_threshold[0]
        endeff_threshold = env.unwrapped.indicator_threshold[1]

        # if endeff_distance < endeff_threshold and angle_distance < angle_threshold:
        if info['angle_success']:
            ob, rew, done, info = env.step(np.asarray([0., 0., 0.]))
            goal_examples.append(ob)
            n += 1
            save_goal_example(env, n)

    assert len(goal_examples) == total_goal_examples, 'Could not generate enough goal examples'
    goal_examples = {
        key: np.concatenate([
            goal_example[key][None] for goal_example in goal_examples
        ], axis=0)
        for key in goal_examples[0].keys()
    }

    return goal_examples

from .goal_collection.dclaw import (
    generate_lift_dd_goals,
    generate_translate_puck_goals)

DEFAULT_TASK_KEY = '__DEFAULT__'
SUPPORTED_ENVS_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Point2D': {DEFAULT_TASK_KEY: generate_point_2d_goals},
        'Pusher2D': {DEFAULT_TASK_KEY: generate_pusher_2d_goals},
        'DClaw': {
            DEFAULT_TASK_KEY: None,
            'LiftDDFixed-v0': generate_lift_dd_goals,
            'TranslatePuckFixed-v0': generate_translate_puck_goals,
        }
    }
}
