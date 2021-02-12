from collections import OrderedDict
from copy import deepcopy

import tensorflow as tf

from softlearning.utils.tensorflow import nest
from softlearning.preprocessors.utils import get_preprocessor_from_params


def get_embedding_from_variant(variant, env, *args, **kwargs):
    from softlearning.models.ddl.distance_estimator import (
        create_embedding_fn)

    distance_fn_params = deepcopy(variant['distance_fn_params'])
    distance_fn_kwargs = deepcopy(distance_fn_params['kwargs'])

    observation_preprocessors_params = distance_fn_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = distance_fn_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value)
        for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    input_shapes = observation_shapes

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    preprocessors = observation_preprocessors

    assert 'embedding_dim' in distance_fn_kwargs, (
        'Must specify an embedding dimension in the distance function kwargs')
    embedding_dim = distance_fn_kwargs.pop('embedding_dim')

    embedding_fn = create_embedding_fn(
        input_shapes=input_shapes,
        embedding_dim=embedding_dim,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **distance_fn_kwargs,
        **kwargs)
    return embedding_fn


def get_distance_estimator_from_variant(variant, env, *args, **kwargs):
    from softlearning.models.ddl.distance_estimator import (
        create_distance_estimator)

    distance_fn_params = deepcopy(variant['distance_fn_params'])
    distance_fn_kwargs = deepcopy(distance_fn_params['kwargs'])

    observation_preprocessors_params = distance_fn_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = distance_fn_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value)
        for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    input_shapes = {
        's1': observation_shapes,
        's2': observation_shapes,
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    preprocessors = {
        's1': observation_preprocessors,
        's2': observation_preprocessors,
    }

    distance_fn = create_distance_estimator(
        input_shapes=input_shapes,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **distance_fn_kwargs,
        **kwargs)
    return distance_fn


def get_vae(encoder_path=None, decoder_path=None, **kwargs):
    from softlearning.models.vae import VAE
    assert encoder_path is not None and decoder_path is not None, (
        "Must specify paths for the encoder/decoder models.")
    vae = VAE(**kwargs)
    vae.encoder.load_weights(encoder_path)
    vae.decoder.load_weights(decoder_path)
    vae.encoder.trainable = False
    vae.decoder.trainable = False
    return vae


def get_dynamics_model_from_variant(variant, env, *args, **kwargs):
    from .dynamics_model import create_dynamics_model

    dynamics_model_params = deepcopy(variant['dynamics_model_params'])
    # dynamics_model_type = deepcopy(dynamics_model_params['type'])
    dynamics_model_kwargs = deepcopy(dynamics_model_params['kwargs'])

    observation_preprocessors_params = dynamics_model_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = dynamics_model_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    encoder_kwargs = dynamics_model_kwargs.pop('encoder_kwargs', {}).copy()
    decoder_kwargs = dynamics_model_kwargs.pop('decoder_kwargs', {}).copy()
    dynamics_latent_dim = dynamics_model_kwargs.pop('dynamics_latent_dim', 16)

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))
    action_shape = env.action_shape

    input_shapes = {
        'observations': observation_shapes,
        'actions': action_shape,
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue

        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    action_preprocessor = None
    preprocessors = {
        'observations': observation_preprocessors,
        'actions': action_preprocessor,
    }

    dynamics_model = create_dynamics_model(
        input_shapes=input_shapes,
        dynamics_latent_dim=dynamics_latent_dim,
        *args,
        observation_keys=observation_keys,
        preprocessors=preprocessors,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        **dynamics_model_kwargs,
        **kwargs)

    return dynamics_model


def get_reward_classifier_from_variant(variant, env, *args, **kwargs):
    from .vice_models import create_feedforward_reward_classifier_function

    reward_classifier_params = deepcopy(variant['reward_classifier_params'])
    reward_classifier_type = deepcopy(reward_classifier_params['type'])
    assert reward_classifier_type == 'feedforward_classifier', (
        reward_classifier_type)
    reward_classifier_kwargs = deepcopy(reward_classifier_params['kwargs'])

    observation_preprocessors_params = reward_classifier_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = reward_classifier_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    # TODO: Clean this up
    dynamics_aware = variant['algorithm_params']['type'] == 'VICEDynamicsAware'

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    if dynamics_aware:
        dynamics_model_kwargs = deepcopy(variant['dynamics_model_params']['kwargs'])
        dynamics_latent_dim = dynamics_model_kwargs['dynamics_latent_dim']
        dynamics_features_shape = tf.TensorShape(dynamics_latent_dim)
        input_shapes = {
            'observations': observation_shapes,
            'dynamics_features': dynamics_features_shape
        }
    else:
        input_shapes = observation_shapes
    
    # from IPython import embed; embed()

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue

        preprocessor_type = preprocessor_params.get('type')
        if preprocessor_type == 'PickledPreprocessor':
            import pickle
            preprocessor_kwargs = preprocessor_params.pop('kwargs', {})
            assert 'preprocessor_path' in preprocessor_kwargs, (
                'Need to specify a .pkl file to load the preprocessor')
            with open(preprocessor_kwargs['preprocessor_path'], 'rb') as f:
                data = pickle.load(f)
                if 'extract_fn' in preprocessor_kwargs:
                    extract_fn = (variant['reward_classifier_params']
                                        ['kwargs']
                                        ['observation_preprocessors_params']
                                        [name]
                                        ['kwargs'].pop('extract_fn'))
                    # extract_fn = preprocessor_kwargs.pop('extract_fn')
                    preprocessor = extract_fn(data)
                else:
                    preprocessor = data
                if isinstance(preprocessor, tf.keras.Model):
                    preprocessor.trainable = False
                observation_preprocessors[name] = preprocessor

        elif preprocessor_type == 'EmbeddingPreprocessor':
            preprocessor_kwargs = preprocessor_params.pop('kwargs', {})
            observation_preprocessors[name] = get_embedding_from_variant(
                variant, env)

        else:
            observation_preprocessors[name] = get_preprocessor_from_params(
                env, preprocessor_params)

    if dynamics_aware:
        preprocessors = {
            'observations': observation_preprocessors,
            'dynamics_features': None,
        }
    else:
        preprocessors = observation_preprocessors

    reward_classifier = create_feedforward_reward_classifier_function(
        input_shapes=input_shapes,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **reward_classifier_kwargs,
        **kwargs)

    return reward_classifier


def get_inputs_for_nested_shapes(input_shapes, name=None):
    if isinstance(input_shapes, dict):
        return type(input_shapes)([
            (name, get_inputs_for_nested_shapes(value, name))
            for name, value in input_shapes.items()
        ])
    elif isinstance(input_shapes, (tuple, list)):
        if all(isinstance(x, int) for x in input_shapes):
            return tf.keras.layers.Input(shape=input_shapes, name=name)
        else:
            return type(input_shapes)((
                get_inputs_for_nested_shapes(input_shape, name=None)
                for input_shape in input_shapes
            ))
    elif isinstance(input_shapes, tf.TensorShape):
        return tf.keras.layers.Input(shape=input_shapes, name=name)

    raise NotImplementedError(input_shapes)


def flatten_input_structure(inputs):
    inputs_flat = nest.flatten(inputs)
    return inputs_flat


def create_input(name, input_shape):
    input_ = tf.keras.layers.Input(
        shape=input_shape,
        name=name,
        dtype=(tf.uint8  # Image observation
               if len(input_shape) == 3 and input_shape[-1] in (1, 3)
               else tf.float32)  # Non-image
    )
    return input_


def create_inputs(input_shapes):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s.

    TODO: Need to figure out a better way for handling the dtypes.
    """
    inputs = nest.map_structure_with_paths(create_input, input_shapes)
    inputs_flat = flatten_input_structure(inputs)

    return inputs_flat
