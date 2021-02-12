from . import (
    simple_replay_pool,
    union_pool,
    active_replay_pool,
    goal_replay_pool,
    hindsight_experience_replay_pool,
    prioritized_experience_replay_pool,
    multigoal_replay_pool,
    partial_save_pool,
    uniformly_reweighted_replay_pool,
)


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'ActiveReplayPool': active_replay_pool.ActiveReplayPool,
    'GoalReplayPool': goal_replay_pool.GoalReplayPool,
    'UnionPool': union_pool.UnionPool,
    'HindsightExperienceReplayPool': (
        hindsight_experience_replay_pool.HindsightExperienceReplayPool),
    'PrioritizedExperienceReplayPool': prioritized_experience_replay_pool.PrioritizedExperienceReplayPool,
    'MultiGoalReplayPool': multigoal_replay_pool.MultiGoalReplayPool,
    'PartialSaveReplayPool': partial_save_pool.PartialSaveReplayPool,
    'UniformlyReweightedReplayPool': uniformly_reweighted_replay_pool.UniformlyReweightedReplayPool,
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_params(replay_pool_params, env, *args, **kwargs):
    replay_pool_type = replay_pool_params.pop('type', DEFAULT_REPLAY_POOL)
    replay_pool_kwargs = replay_pool_params.pop('kwargs', {})

    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        environment=env,
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool


def get_replay_pool_from_variant(variant, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params'].copy()
    replay_pool = get_replay_pool_from_params(
        replay_pool_params, *args, **kwargs)

    return replay_pool
