from .adapters.gym_adapter import GymAdapter

try:
    import gym_minigrid
except:
    print('Warning: gym_minigrid package not found.')

ADAPTERS = {
    'gym': GymAdapter,
}

try:
    from .adapters.dm_control_adapter import DmControlAdapter
    ADAPTERS['dm_control'] = DmControlAdapter
except ModuleNotFoundError as e:
    if 'dm_control' not in e.msg:
        raise

    print("Warning: dm_control package not found. Run"
          " `pip install git+https://github.com/deepmind/dm_control.git`"
          " to use dm_control environments.")

try:
    from .adapters.robosuite_adapter import RobosuiteAdapter
    ADAPTERS['robosuite'] = RobosuiteAdapter
except ModuleNotFoundError as e:
    if 'robosuite' not in e.msg:
        raise

    print("Warning: robosuite package not found. Run `pip install robosuite`"
          " to use robosuite environments.")

try:
    import multiworld
    multiworld.register_all_envs()
except ModuleNotFoundError as e:
    if 'multiworld' not in e.msg:
        raise
    print("Warning: multiworld package not found.")


UNIVERSES = set(ADAPTERS.keys())

def get_environment(universe, domain, task, environment_kwargs):
    return ADAPTERS[universe](
            domain,
            task,
            **environment_kwargs)

def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    # from IPython import embed; embed()

    return get_environment(universe, domain, task, environment_kwargs)

def get_goal_example_environment_from_variant(environment_name, gym_adapter=True, eval=False):
    import gym

    if environment_name not in [env.id for env  in gym.envs.registry.all()]:
        from multiworld.envs.mujoco import register_goal_example_envs
        register_goal_example_envs()

    if gym_adapter:
        return GymAdapter(env=gym.make(environment_name))
    else:
        return gym.make(environment_name)
