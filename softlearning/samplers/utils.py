from collections import defaultdict

import numpy as np

from softlearning import replay_pools
from . import (
    dummy_sampler,
    remote_sampler,
    base_sampler,
    simple_sampler,
    active_sampler,
    goal_sampler,
    pool_sampler,
    nn_sampler,
    classifier_sampler)


def get_sampler_from_variant(variant, *args, **kwargs):
    SAMPLERS = {
        'DummySampler': dummy_sampler.DummySampler,
        'RemoteSampler': remote_sampler.RemoteSampler,
        'Sampler': base_sampler.BaseSampler,
        'SimpleSampler': simple_sampler.SimpleSampler,
        'ActiveSampler': active_sampler.ActiveSampler,
        'GoalSampler': goal_sampler.GoalSampler,
        'PoolSampler': pool_sampler.PoolSampler,
        'NNSampler': nn_sampler.NNSampler,
        'PoolSampler': pool_sampler.PoolSampler,
        'ClassifierSampler': classifier_sampler.ClassifierSampler,
    }

    sampler_params = variant['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = sampler_params.get('args', ())
    sampler_kwargs = sampler_params.get('kwargs', {}).copy()

    sampler = SAMPLERS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler


DEFAULT_PIXEL_RENDER_KWARGS = {
    'mode': 'rgb_array',
    'width': 256,
    'height': 256,
}

DEFAULT_HUMAN_RENDER_KWARGS = {
    'mode': 'human',
    'width': 500,
    'height': 500,
}


def rollout(env,
            policy,
            path_length,
            sampler_class=simple_sampler.SimpleSampler,
            algorithm=None,
            extra_fields=None,
            sampler_kwargs=None,
            callback=None,
            render_kwargs=None,
            break_on_terminal=True):
    pool = replay_pools.SimpleReplayPool(
        env, 
        extra_fields=extra_fields, 
        max_size=path_length)

    if sampler_kwargs:
        sampler = sampler_class(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=None,
            **sampler_kwargs)
    else:
        sampler = sampler_class(
            max_path_length=path_length,
            min_pool_size=None,
            batch_size=None)

    if hasattr(sampler, 'set_algorithm'):
        sampler.set_algorithm(algorithm)

    sampler.initialize(env, policy, pool)

    render_mode = (render_kwargs or {}).get('mode', None)
    if render_mode == 'rgb_array':
        render_kwargs = {
            **DEFAULT_PIXEL_RENDER_KWARGS,
            **render_kwargs
        }
    elif render_mode == 'human':
        render_kwargs = {
            **DEFAULT_HUMAN_RENDER_KWARGS,
            **render_kwargs
        }
    else:
        render_kwargs = None

    images = []
    infos = defaultdict(list)
    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        for key, value in info.items():
            infos[key].append(value)

        if callback is not None:
            callback(observation)

        if render_kwargs:
            if render_mode == 'rgb_array':
                #note: this will only work for mujoco-py environments
                if hasattr(env.unwrapped, 'imsize'):
                    imsize = env.unwrapped.imsize
                else:
                    imsize = 200

                imsize_flat = imsize*imsize*3
                #for goal conditioned stuff
                #if observation['observations'].shape[0] == 2*imsize_flat:
                #    image1 = observation['observations'][:imsize_flat].reshape(48,48,3)
                #    image2 = observation['observations'][imsize_flat:].reshape(48,48,3)
                #    image1 = (image1*255.0).astype(np.uint8)
                #    image2 = (image2*255.0).astype(np.uint8)
                #    image = np.concatenate([image1, image2], axis=1)

                if 'pixels' in observation.keys() and observation['pixels'].shape[-1] == 6:
                    pixels = observation['pixels']
                    image1 = pixels[:, :, :3]
                    image2 = pixels[:, :, 3:]
                    image = np.concatenate([image1, image2], axis=1)
                else:
                    image = env.render(**render_kwargs)
                images.append(image)
            else:
                image = env.render(**render_kwargs)
                images.append(image)

        if terminal:
            policy.reset()
            if break_on_terminal: break

    assert pool._size == t + 1

    path = pool.batch_by_indices(np.arange(pool._size))
    path['infos'] = infos

    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(n_paths, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_paths)]
    return paths
