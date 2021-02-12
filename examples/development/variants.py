from copy import deepcopy

from ray import tune
import numpy as np
import os
from softlearning.misc.utils import get_git_rev, deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

# M = number of hidden units per layer
# N = number of hidden layers
# M = 512
# N = 2
M = 256
N = 2

REPARAMETERIZE = True
NUM_COUPLING_LAYERS = 2


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'squash': True,
        'observation_keys': None,
        'goal_keys': None,
        'observation_preprocessors_params': {}
    }
}


ALGORITHM_PARAMS_BASE = {
    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_n_episodes': 3,
        'eval_deterministic': False,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'save_training_video_frequency': 5,
        'eval_render_kwargs': {
            'width': 480,
            'height': 480,
            'mode': 'rgb_array',
        },
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'n_initial_exploration_steps': int(1e3),
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'verbose': True,

            'eval_n_episodes': 3,

            'ext_reward_coeff': 1,
            'rnd_int_rew_coeff': tune.grid_search([0]),
            'normalize_ext_reward_gamma': 0.99,
        },
        # 'rnd_params': {
        #     'convnet_params': {
        #         'conv_filters': (16, 32, 64),
        #         'conv_kernel_sizes': (3, 3, 3),
        #         'conv_strides': (2, 2, 2),
        #         'normalization_type': None,
        #     },
        #     'fc_params': {
        #         'hidden_layer_sizes': (256, 256),
        #         'output_size': 512,
        #     },
        # }
    },
    'MultiSAC': {
        'type': 'MultiSAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            # 'n_initial_exploration_steps': int(1e4),
            'n_initial_exploration_steps': int(5e3),
            'action_prior': 'uniform',
            'her_iters': tune.grid_search([0]),
            'rnd_int_rew_coeffs': [0, 0], # [1, 1],
            'ext_reward_coeffs': [1, 1], # 0 corresponds to reset policy
            'normalize_ext_reward_gamma': 0.99,
            'share_pool': False,
        },
        'rnd_params': {
            'convnet_params': {
                'conv_filters': (16, 32, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
            },
            'fc_params': {
                'hidden_layer_sizes': (256, 256),
                'output_size': 512,
            },
        },
    },
    'HERQLearning': {
        'type': 'HERQLearning',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'n_initial_exploration_steps': int(5e3),
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'ext_reward_coeff': 1,
            'eval_n_episodes': 3,
            'rnd_int_rew_coeff': tune.grid_search([0]),
            # 'normalize_ext_reward_gamma': 0.99,
            'verbose': True,

            'replace_original_reward': tune.grid_search([True, False]), # True,
        },
    },

}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 100,
    'gym': {
        DEFAULT_KEY: 100,
        'Point2D': {
            DEFAULT_KEY: 200,
        },
        'Pusher2D': {
            DEFAULT_KEY: 100,
            'Simple-v0': 150,
            'Test-v0': 150,
        },
        'MiniGrid': {
            DEFAULT_KEY: 50,
        },
        'DClaw': {
            DEFAULT_KEY: 50,
            'TurnFixed-v0': 50,
            # 'TurnResetFree-v0': 100,
            'TurnResetFree-v0': 50,
            'TurnResetFreeSwapGoal-v0': tune.grid_search([100]),
            'TurnResetFreeRandomGoal-v0': 100,
            'TurnFreeValve3Fixed-v0': tune.grid_search([50]),
            # 'TurnFreeValve3RandomReset-v0': 50,
            'TurnFreeValve3ResetFree-v0': tune.grid_search([100]),
            'TurnFreeValve3ResetFreeSwapGoal-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeComposedGoals-v0': tune.grid_search([150]),

            # Translating Tasks
            'TranslatePuckFixed-v0': 50,
            'TranslateMultiPuckFixed-v0': 100,

            'TranslatePuckResetFree-v0': 50,

            # Lifting Tasks
            'LiftDDFixed-v0': tune.grid_search([50]),
            'LiftDDResetFree-v0': tune.grid_search([50]),

            # Flipping Tasks
            'FlipEraserFixed-v0': tune.grid_search([50]),
            'FlipEraserResetFree-v0': tune.grid_search([50]),
            'FlipEraserResetFreeSwapGoal-v0': tune.grid_search([50]),

            # Sliding Tasks
            'SlideBeadsFixed-v0': tune.grid_search([25]),
            'SlideBeadsResetFree-v0': tune.grid_search([25]),
            'SlideBeadsResetFreeEval-v0': tune.grid_search([25]),
        },
    },
}


NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 200,
    'gym': {
        DEFAULT_KEY: 200,
        'Point2D': {
            DEFAULT_KEY: int(300),
        },
        'Pusher2D': {
            DEFAULT_KEY: int(100),
        },
        'MiniGrid': {
            DEFAULT_KEY: 100,
        },
        'DClaw': {
            DEFAULT_KEY: int(250),
            'TurnFreeValve3Fixed-v0': 750,
            'TranslateMultiPuckFixed-v0': 500,
        },
    },
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        'Point2D': {
            # === Point Mass ===
            'Fixed-v0': {
                # 'boundary_distance': tune.grid_search([8, 16]),
                # 'action_scale': tune.grid_search([0.5, 0.25]),
                'action_scale': 0.5,
                'images_are_rgb': True,
                'init_pos_range': None,   # Random reset
                'target_pos_range': None, # Random target
                'render_onscreen': False,
                # 'reward_type': tune.grid_search(['dense', 'sparse']),
                'reward_type': tune.grid_search(['sparse']),
                'observation_keys': ('state_achieved_goal', 'state_desired_goal'),
                # 'goal_keys': ('state_desired_goal', ),
            },
            'SingleWall-v0': {
                # 'boundary_distance': tune.grid_search([4, 8]),
                'action_scale': tune.grid_search([1.0, 0.5]),
                'images_are_rgb': True,
                'init_pos_range': None,   # Random reset
                'target_pos_range': None, # Random target
                'render_onscreen': False,
                'reward_type': tune.grid_search(['dense', 'sparse']),
                'observation_keys': ('state_observation', 'state_desired_goal'),
                # 'goal_keys': ('state_desired_goal', ),
            },
            'BoxWall-v1': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                'reward_type': tune.grid_search(['sparse']),
                'init_pos_range': ((-3, -3), (-3, -3)),
                # 'init_pos_range': None,   # Random reset
                'target_pos_range': ((3, 3), (3, 3)),
                'render_onscreen': False,
                'observation_keys': ('state_achieved_goal', 'state_desired_goal'),
            },
            'Maze-v0': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                'reward_type': tune.grid_search(['sparse']),
                'render_onscreen': False,
                'observation_keys': ('state_achieved_goal', 'state_desired_goal'),
                'use_count_reward': tune.grid_search([True, False]),
                'n_bins': 10,

                # === EASY ===
                # 'wall_shape': 'easy-maze',
                # 'init_pos_range': ((-2.5, -3), (-2.5, -3)),
                # 'target_pos_range': ((2.5, -3), (2.5, -3)),
                # === MEDIUM ===
                'wall_shape': 'medium-maze',
                'init_pos_range': ((-3, -3), (-3, -3)),
                'target_pos_range': ((3, 3), (3, 3)),
                # === HARD ===
                # 'wall_shape': 'hard-maze',
                # 'init_pos_range': ((-3, -3), (-3, -3)),
                # 'target_pos_range': ((-0.5, 1.25), (-0.5, 1.25)),
            },

#             'Fixed-v1': {
#                 'ball_radius': 0.5,
#                 'target_radius': 0.5,
#                 'boundary_distance': 4,
#                 'images_are_rgb': True,
#                 'init_pos_range': None,
#                 'target_pos_range': None,
#                 'render_onscreen': False,
#                 'reward_type': 'sparse',
#                 'observation_keys': ('state_observation', ),
#                 'goal_keys': ('state_desired_goal', ),
#             },
        },
        'Pusher2D': {
            'Simple-v0': {
                'init_qpos_range': ((0, 0, 0), (0, 0, 0)),
                'init_object_pos_range': ((1, 0), (1, 0)),
                'target_pos_range': ((2, 2), (2, 2)),
                'reset_gripper': True,
                'reset_object': True,
                'observation_keys': (
                    # 'observation',
                    'gripper_qpos',
                    'gripper_qvel',
                    'object_pos',
                    'object_vel',
                    'target_pos',
                ),
            },
            'Test-v0': {
                'do_reset': True,
                'multi_reset': False,
                'multi_reset_block': False,
                'reset_block': True,
                'reset_gripper': True,
            }
        },
        'DClaw': {
            # === Fixed Screw ===
            'TurnFixed-v0': {
                'reward_keys_and_weights': {
                    # 'object_to_target_angle_distance_reward': 1,
                    'sparse_reward': 1,
                },
                'init_pos_range': (0, 0),
                'target_pos_range': (np.pi, np.pi),
                'observation_keys': (
                    'object_angle_cos',
                    'object_angle_sin',
                    'claw_qpos',
                    'last_action'
                ),
            },

            'PoseStatic-v0': {},
            'PoseDynamic-v0': {},
            'TurnRandom-v0': {},
            'TurnResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                'target_pos_range': (np.pi, np.pi),
            },
            'TurnResetFreeSwapGoal-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'reset_fingers': True,
            },
            'TurnResetFreeRandomGoal-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'reset_fingers': True,
            },
            'TurnRandomDynamics-v0': {},
            'TurnFreeValve3Fixed-v0': {
                'reward_keys_and_weights': {
                    # 'object_to_target_position_distance_reward': tune.grid_search([2]),
                    # 'object_to_target_orientation_distance_reward': 1,
                    'sparse_reward': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
                'init_qpos_range': ((0, 0, 0, 0, 0, 0), ) * 2,
                'target_qpos_range': ((0, 0, 0, 0, 0, np.pi), ) * 2,
                # 'target_qpos_range': [
                #     (0.01, 0.01, 0, 0, 0, -np.pi / 2),
                #     (-0.01, -0.01, 0, 0, 0, np.pi / 2)
                # ],
                # 'init_qpos_range': (
                #     (-0.08, -0.08, 0, 0, 0, -np.pi),
                #     (0.08, 0.08, 0, 0, 0, np.pi)
                # ),
            },
            'TurnFreeValve3ResetFree-v0': {
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'reset_frequency': 0,
                'target_qpos_range': [
                    (0.01, 0.01, 0, 0, 0, -np.pi / 2),
                    (-0.01, -0.01, 0, 0, 0, np.pi / 2)
                ],
                'init_qpos_range': [
                    (0, 0, 0, 0, 0, 0)
                ],
                # === BELOW IS FOR SAVING INTO THE REPLAY POOL. ===
                # MAKE SURE TO SET `no_pixel_information = True` below in order
                # to remove the pixels from the policy inputs/Q inputs.
                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #     },
                # },
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.38,
                #     'elevation': -36,
                #     'lookat': (0.04, 0.008, 0.025),
                # },
            },
            'TurnFreeValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'initial_distribution_path': '',
                'reset_from_corners': True,
            },
            'TurnFreeValve3ResetFreeRandomGoal-v0': {
                'observation_keys': (
                    'claw_qpos',
                    'object_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                    'last_action',
                    'target_orientation',
                    'object_to_target_relative_position',
                ),
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'reset_fingers': True,
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_orientation_distance_reward': 1,
                    # 'object_to_target_position_distance_reward': tune.grid_search([1]),
                    # 'object_to_target_orientation_distance_reward': 0,
                },
                'reset_fingers': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
                'goals': tune.grid_search([
                    [(0, 0, 0, 0, 0, np.pi / 2), (-0.05, -0.06, 0, 0, 0, 0)],
                    # [(0.05, 0.06, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                    # [(0, 0, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                ]),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
                'reward_keys_and_weights': {
                    # 'object_to_target_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_position_distance_reward': tune.grid_search([2]),
                    'object_to_target_orientation_distance_reward': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    'target_xy_position',
                ),
                # 'goals': tune.grid_search([
                #     [(0, 0, 0, 0, 0, np.pi / 2), (-0.05, -0.06, 0, 0, 0, 0)],
                #     [(0.05, 0.06, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                #     [(0, 0, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                # ]),
            },
            'TurnFreeValve3ResetFreeCurriculum-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'reset_fingers': False,
            },
            'XYTurnValve3Fixed-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
            },
            'XYTurnValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
                'num_goals': 1,
            },
            'XYTurnValve3Random-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
            },
            'XYTurnValve3ResetFree-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
                'reset_fingers': tune.grid_search([True, False]),
                'reset_arm': False,
            },
            # Lifting Tasks
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0, #5,
                },
                'init_qpos_range': (
                    (-0.05, -0.05, 0.041, -np.pi, -np.pi, -np.pi),
                    (0.05, 0.05, 0.041, np.pi, np.pi, np.pi)
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
            },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 0,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
            },

            # Flipping Tasks
            'FlipEraserFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                'target_qpos_range': [(0, 0, 0, np.pi, 0, 0)],
            },
            'FlipEraserResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
            },
            'FlipEraserResetFreeSwapGoal-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
            },

            # === Translation Tasks ===
            'TranslateMultiPuckFixed-v0': {
                'init_qpos_ranges': (
                    ((0.1, 0.1, 0, 0, 0, 0), (0.1, 0.1, 0, 0, 0, 0)),
                    ((-0.1, -0.1, 0, 0, 0, 0), (-0.1, -0.1, 0, 0, 0, 0)),
                ),
                'target_qpos_ranges': (
                    ((0.1, -0.1, 0, 0, 0, 0), (0.1, -0.1, 0, 0, 0, 0)),
                    ((-0.1, 0.1, 0, 0, 0, 0), (-0.1, 0.1, 0, 0, 0, 0)),
                ),
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object1_xy_position',
                    'object2_xy_position',
                ),
                'reward_keys_and_weights': {
                    'object1_to_target_position_distance_log_reward': 1,
                    'object2_to_target_position_distance_log_reward': 1,
                }
            }

        }
    }
}


FREE_SCREW_VISION_KWARGS = {
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
            'width': 32,
            'height': 32,
            'camera_id': -1,
        },
    },
    'camera_settings': {
        'azimuth': 180,
        'distance': 0.38,
        'elevation': -36,
        'lookat': (0.04, 0.008, 0.026),
    },
}
FIXED_SCREW_VISION_KWARGS = {
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
           'width': 32,
           'height': 32,
           'camera_id': -1,
        }
    },
    'camera_settings': {
        'azimuth': 180,
        'distance': 0.3,
        'elevation': -50,
        'lookat': np.array([0.02, 0.004, 0.09]),
    },
}
SLIDE_BEADS_VISION_KWARGS = {
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
            'width': 32,
            'height': 32,
            'camera_id': -1,
        },
    },
    'camera_settings': {
        'azimuth': 90,
        'distance': 0.37,
        'elevation': -45,
        'lookat': (0, 0.0046, -0.016),
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'DClaw': {
            'TurnFixed-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'init_pos_range': (-np.pi, np.pi),
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                    # 'target_angle_cos',
                    # 'target_angle_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            'TurnResetFree-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                    # 'target_angle_cos',
                    # 'target_angle_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            # Free screw
            'TurnFreeValve3Fixed-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, -np.pi),
                    (0.08, 0.08, 0, 0, 0, np.pi)
                ),
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                #     (0, 0, 0, 0, 0, -np.pi / 2)
                # ],
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2)
                ],
            },
            # === Reset-free environment below ===
            'TurnFreeValve3ResetFree-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'reset_frequency': 0,
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    # (0, 0, 0, 0, 0, -np.pi / 2)
                    (0, 0, 0, 0, 0, np.pi / 2)
                ],
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([0.5, 1]),
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, -np.pi),
                    (0.08, 0.08, 0, 0, 0, np.pi)
                ),
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                ),
            },
            'TurnFreeValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'initial_distribution_path': '',
                'reset_from_corners': True,
            },
            'ScrewFixed-v0': {},
            'ScrewRandom-v0': {},
            'ScrewRandomDynamics-v0': {},
            # Translating Puck Tasks
            'TranslatePuckFixed-v0': {
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0)
                ],
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, 0),
                    (0.08, 0.08, 0, 0, 0, 0)
                ),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                },
            },
            'TranslatePuckResetFree-v0': {
                'target_qpos_range': [
                    (-0.08, -0.08, 0, 0, 0, 0),
                    (0.08, 0.08, 0, 0, 0, 0)
                ],
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                },
            },

            # Lifting Tasks
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 5,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi)
                ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'use_bowl_arena': False,
            },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 1,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 64,
                        'height': 64,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'object_position',
                    'object_quaternion',
                    'last_action',
                    'target_position',
                    'target_quaternion',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.26,
                    'elevation': -40,
                    'lookat': (0, 0, 0.06),
                }
            },
            'LiftDDResetFreeComposedGoals-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 1,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_policy_checkpoint_path': '',
                'goals': [
                     (0, 0, 0, 0, 0, 0),
                     (0, 0, 0.05, 0, 0, 0),
                 ],
                'reset_frequency': 0,
            },
            # Flipping Tasks
            'FlipEraserFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
                # In bowl
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.26,
                #     'elevation': -32,
                #     'lookat': (0, 0, 0.06)
                # },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },
            'LiftDDResetFree-v0': {
                # For repositioning
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 0,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
                # For Lifting
                # 'reward_keys_and_weights': {
                #     'object_to_target_z_position_distance_reward': 10,
                #     'object_to_target_xy_position_distance_reward': tune.grid_search([1, 2]),
                #     'object_to_target_orientation_distance_reward': 0,
                # },
                # 'init_qpos_range': (
                #     (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                #     (0, 0, 0.041, np.pi, np.pi, np.pi),
                # ),
                # 'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1,
                    }
                },
                # In box
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.35,
                    'elevation': -55,
                    'lookat': (0, 0, 0.03)
                },
                # In bowl
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.26,
                #     'elevation': -32,
                #     'lookat': (0, 0, 0.06)
                # },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },
            # Sliding Tasks
            'SlideBeadsFixed-v0': {
                **SLIDE_BEADS_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': (
                    (-0.0475, -0.0475, -0.0475, -0.0475),
                    (0.0475, 0.0475, 0.0475, 0.0475),
                ),
                'target_qpos_range': [
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'num_objects': 4,
                'cycle_goals': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    # === BELOW JUST FOR LOGGING == 
                    'objects_target_positions',
                    'objects_positions',
                ),
            },
            'SlideBeadsResetFree-v0': {
                **SLIDE_BEADS_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                    # 'objects_to_targets_mean_distance_reward': 0, # Make sure 0 ext reward
                },
                'init_qpos_range': [(0, 0, 0, 0)],
                # LNT Baseline 
                # 'target_qpos_range': [
                    # (0, 0, 0, 0),
                    # (-0.0475, -0.0475, 0.0475, 0.0475),
                # ],
                # 1 goal with RND reset controller
                'target_qpos_range': [
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'num_objects': 4,
                'cycle_goals': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    # === BELOW JUST FOR LOGGING ===
                    'objects_positions',
                    'objects_target_positions',
                ),
            },
            'SlideBeadsResetFreeEval-v0': {
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': [(0, 0, 0, 0)],
                'num_objects': 4,
                'target_qpos_range': [
                    (0, 0, 0, 0),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                # 'target_qpos_range': [
                #     (0, 0),
                #     (-0.0825, 0.0825),
                #     (0.0825, 0.0825),
                #     (-0.04, 0.04),
                #     (-0.0825, -0.0825),
                # ],
                'cycle_goals': True,
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'objects_positions',
                    'last_action',
                    'objects_target_positions',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 23.234042553191497,
                    'distance': 0.2403358053524018,
                    'elevation': -29.68085106382978,
                    'lookat': (-0.00390331,  0.01236683,  0.01093447),
                }
            },
        },
    },
}


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 50 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_algorithm_params(universe, domain, task):
    algorithm_params = {
        'kwargs': {
            'n_epochs': get_num_epochs(universe, domain, task),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task, from_vision):
    if from_vision:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


NUM_CHECKPOINTS = 10
SAMPLER_PARAMS_PER_DOMAIN = {
    'DClaw': {
        'type': 'SimpleSampler',
    },
}


def get_variant_spec_base(universe, domain, task, task_eval, policy, algorithm, from_vision):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        get_algorithm_params(universe, domain, task),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task, from_vision),
            },
            'evaluation': {
                'domain': domain,
                'task': task_eval,
                'universe': universe,
                'kwargs': (
                    tune.sample_from(lambda spec: (
                        spec.get('config', spec)
                        ['environment_params']
                        ['training']
                        .get('kwargs')
                    ))
                    if task == task_eval
                    else get_environment_params(universe, domain, task_eval, from_vision)),
            },
        },
        'policy_params': get_policy_params(universe, domain, task),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, ) * N,
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'observation_preprocessors_params': {}
            },
            # 'discrete_actions': False,
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(5e5),
            },
            # 'type': 'HindsightExperienceReplayPool',
            # 'kwargs': {
            #     'max_size': int(5e5),
            #     'her_strategy':{
            #         'resampling_probability': 0.5,
            #         'type': 'final',
            #     }
            # },
        },
        'sampler_params': deep_update({
            # 'type': 'GoalSampler',
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['sampler_params']['kwargs']['max_path_length']
                )),
                'batch_size': 256, # tune.grid_search([128, 256]),
                'store_last_n_paths': 20,
            }
        }, SAMPLER_PARAMS_PER_DOMAIN.get(domain, {})),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    # Set this flag if you don't want to pass pixels into the policy/Qs
    no_pixel_information = False

    # TODO: Clean this up
    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    env_obs_keys = env_kwargs.get('observation_keys', tuple())
    env_goal_keys = env_kwargs.get('goal_keys', tuple())

    # === FROM VISION ===
    if from_vision and "pixel_wrapper_kwargs" in env_kwargs.keys() and \
       "device_path" not in env_kwargs.keys():
        # === COMMENT BELOW TO SAVE PIXELS INTO POOL ===
        obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        non_image_obs_key = vsariant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys
        # == FILTER OUT GROUND TRUTH STATE ===
        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_object_obs_keys
    # === FROM STATE / NO PIXEL INFORMATION ===
    elif no_pixel_information or not from_vision:
        non_pixel_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_pixel_obs_keys

    if env_goal_keys:
        variant_spec['policy_params']['kwargs']['goal_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['goal_keys'] = variant_spec[
                'Q_params']['kwargs']['goal_keys'] = env_goal_keys

    if 'ResetFree' not in task:
        variant_spec['algorithm_params']['kwargs']['save_training_video_frequency'] = 0

    if domain == 'MiniGrid':
        variant_spec['algorithm_params']['kwargs']['reparameterize'] = False
        variant_spec['policy_params']['type'] = 'DiscretePolicy'
        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (32, 32)
        variant_spec['exploration_policy_params']['type'] = 'UniformDiscretePolicy'
        variant_spec['environment_params']['training']['kwargs']['normalize'] = False

    return variant_spec


IMAGE_ENVS = (
    ('robosuite', 'InvisibleArm', 'FreeFloatManipulation'),
)

def is_image_env(universe, domain, task, variant_spec):

    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs'])
            or (universe, domain, task) in IMAGE_ENVS)


STATE_PREPROCESSOR_PARAMS = {
    'ReplicationPreprocessor': {
        'type': 'ReplicationPreprocessor',
        'kwargs': {
            'n': 0,
            'scale_factor': 1,
        }
    },
    'RandomNNPreprocessor': {
        'type': 'RandomNNPreprocessor',
        'kwargs': {
            'hidden_layer_sizes': (32, 32),
            'activation': 'linear',
            'output_activation': 'linear',
        }
    },
    'RandomMatrixPreprocessor': {
        'type': 'RandomMatrixPreprocessor',
        'kwargs': {
            'output_size_scale_factor': 1,
            'coefficient_range': (-1., 1.),
        }
    },
    'None': None,
}


from softlearning.misc.utils import PROJECT_PATH, NFS_PATH
PIXELS_PREPROCESSOR_PARAMS = {
    'StateEstimatorPreprocessor': {
        'type': 'StateEstimatorPreprocessor',
        'kwargs': {
            'input_shape': (32, 32, 3),
            'num_hidden_units': 512,
            'num_hidden_layers': 2,
            'state_estimator_path': '/root/softlearning/softlearning/models/state_estimators/state_estimator_from_vae_latents.h5',
            'preprocessor_params': {
                'type': 'VAEPreprocessor',
                'kwargs': {
                    'encoder_path': '/root/softlearning/softlearning/models/vae_16_dim_beta_3_invisible_claw_l2_reg/encoder_16_dim_3.0_beta.h5',
                    'decoder_path': '/root/softlearning/softlearning/models/vae_16_dim_beta_3_invisible_claw_l2_reg/decoder_16_dim_3.0_beta.h5',
                    'trainable': False,
                    'image_shape': (32, 32, 3),
                    'latent_dim': 16,
                    'include_decoder': False,
                }
            }
        }
    },
    'VAEPreprocessor': {
        'type': 'VAEPreprocessor',
        'kwargs': {
            # 'image_shape': (32, 32, 3),
            'image_shape': (64, 64, 3),
            # 'latent_dim': 16,
            'latent_dim': 64,
            'encoder_path': os.path.join(NFS_PATH,
                                        'pretrained_models',
                                        'vae_64_dim_beta_5_visible_claw_diff_angle',
                                        'encoder_64_dim_5.0_beta.h5'),
            'trainable': False,
        },
    },
    # TODO: Merge OnlineVAEPreprocessor and VAEPreprocessor, just don't update
    # in SAC if not online
    'OnlineVAEPreprocessor': {
        'type': 'OnlineVAEPreprocessor',
        'kwargs': {
            'image_shape': (32, 32, 3),
            'latent_dim': 16,
            # 'latent_dim': 32,
            'beta': 0.5,
            # 'beta': 1e-5,
            # Optionally specify a pretrained model to start finetuning
            # 'encoder_path': os.path.join(PROJECT_PATH,
            #                              'softlearning',
            #                              'models',
            #                              'free_screw_vae_32_dim',
            #                              'encoder_32_dim_0.5_beta_final.h5'),
            # 'decoder_path': os.path.join(PROJECT_PATH,
            #                              'softlearning',
            #                              'models',
            #                              'free_screw_vae_32_dim',
            #                              'decoder_32_dim_0.5_beta_final.h5'),
        },
        'shared': True,
    },
    'RAEPreprocessor': {
        'type': 'RAEPreprocessor',
        'kwargs': {
            'image_shape': (32, 32, 3),
            'latent_dim': 32,
        },
        'shared': True,
    },
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (8, 16, 32),
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': tune.sample_from([None]),
                'downsampling_type': 'conv',
            },
        }
        # {
        #     'type': 'ConvnetPreprocessor',
        #     'kwargs': {
        #         'conv_filters': (64, ) * 4,
        #         'conv_kernel_sizes': (3, ) * 4,
        #         'conv_strides': (2, ) * 4,
        #         'normalization_type': normalization_type,
        #         'downsampling_type': 'conv',
        #         'output_kwargs': {
        #             'type': 'flatten',
        #         },
        #     },
        # }
        for normalization_type in (None, )
    ]),
}

def get_variant_spec_image(universe,
                           domain,
                           task,
                           task_eval,
                           policy,
                           algorithm,
                           from_vision,
                           preprocessor_type,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe,
        domain,
        task,
        task_eval,
        policy,
        algorithm,
        from_vision,
        *args, **kwargs)

    if from_vision and is_image_env(universe, domain, task, variant_spec):
        assert preprocessor_type in PIXELS_PREPROCESSOR_PARAMS or preprocessor_type is None
        if preprocessor_type is None:
            preprocessor_type = "ConvnetPreprocessor"
        preprocessor_params = PIXELS_PREPROCESSOR_PARAMS[preprocessor_type]

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )
    elif preprocessor_type:
        # Assign preprocessor to all parts of the state
        assert preprocessor_type in STATE_PREPROCESSOR_PARAMS
        preprocessor_params = STATE_PREPROCESSOR_PARAMS[preprocessor_type]
        obs_keys = variant_spec['environment_params']['training']['kwargs'].get('observation_keys', tuple())

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                key: deepcopy(preprocessor_params)
                for key in obs_keys
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task, task_eval = (
        args.universe,
        args.domain,
        args.task,
        args.task_evaluation)

    from_vision = args.vision
    preprocessor_type = args.preprocessor_type

    variant_spec = get_variant_spec_image(
        universe,
        domain,
        task,
        task_eval,
        args.policy,
        args.algorithm,
        from_vision,
        preprocessor_type)

    # if args.checkpoint_replay_pool is not None:
    variant_spec['run_params']['checkpoint_replay_pool'] = (
        args.checkpoint_replay_pool or False)

    return variant_spec
