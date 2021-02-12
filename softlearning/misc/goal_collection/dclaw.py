import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy


def generate_lift_dd_goals(env,
                           num_total_examples=200,
                           rollout_length=15,
                           goal_threshold=None,  # TODO: incorporate threshold
                           include_transitions=True,
                           save_image=True):
    env = deepcopy(env)
    # === Modify the init range to have the DD start in the air ===
    env.unwrapped._initial_claw_qpos = np.array([0, -np.pi / 6, np.pi / 2] * 3)
    env.unwrapped._init_qpos_range = (
        (0, 0, 0.1, -np.pi, -np.pi, -np.pi),
        (0, 0, 0.1, np.pi, np.pi, np.pi)
    )

    env.reset()
    if save_image:
        plt.figure(figsize=(4, 4))
        plt.imshow(env.render(mode='rgb_array'))
        path = os.path.join(os.getcwd(), 'env_frame.jpg')
        plt.savefig(path)

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
        r, t = 0, 0
        while r == 0 and t <= rollout_length:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)

            if r == 0:
                observations.append(prev_obs)
                next_observations.append(obs)
                actions.append(action)
                num_positives += 1

            t += 1
            prev_obs = obs

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

    if include_transitions:
        goal_transitions = {
            'observations': goal_obs,
            'next_observations': goal_next_obs,
            'actions': goal_actions,
        }
        return goal_transitions
    else:
        # Rewards are associated with the state @ the next transition
        return goal_next_obs


def generate_translate_puck_goals(env,
                                  num_total_examples=200,
                                  rollout_length=25,
                                  goal_threshold=0.03,  # TODO: incorporate threshold
                                  include_transitions=True,
                                  save_image=True):
    env = deepcopy(env)
    # === Modify the init range to have the DD start in the air ===
    targetx, targety = env.unwrapped._object_target_position[:2]
    env.unwrapped._init_qpos_range = (
        (targetx, targety, 0, 0, 0, -np.pi),
        (targetx, targety, 0, 0, 0, np.pi)
    )

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
        t = 0
        while t <= rollout_length:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)

            dist_to_target = env.get_obs_dict()['object_to_target_position_distance']
            if dist_to_target <= goal_threshold:
                observations.append(prev_obs)
                next_observations.append(obs)
                actions.append(action)
                num_positives += 1
                if save_image:
                    plt.figure(figsize=(4, 4))
                    plt.imshow(env.render(mode='rgb_array'))
                    path = os.path.join(os.getcwd(), 'env_frame.jpg')
                    plt.savefig(path)
                    save_image = False

            t += 1
            prev_obs = obs

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

    if include_transitions:
        goal_transitions = {
            'observations': goal_obs,
            'next_observations': goal_next_obs,
            'actions': goal_actions,
        }
        return goal_transitions
    else:
        # Rewards are associated with the state @ the next transition
        return goal_next_obs
