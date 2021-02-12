from softlearning.environments.utils import get_environment_from_params
import gym

params = {
    'universe': 'gym',
    'domain': 'Point2D',
    'task': 'Fixed-v0',
    'kwargs': {
        'normalize': False,
        'init_pos_range': ((0, 0), (0, 0)),
        'target_pos_range': ((-2, -2), (2, 2)),
        'observation_keys': ('state_observation', 'state_desired_goal'),
    }
}

env = get_environment_from_params(params)

# for _ in range(100):
env.reset()
for _ in range(10):
    env.step(env.action_space.sample())
    env.render()
