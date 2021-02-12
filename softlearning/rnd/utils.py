import tensorflow as tf
from collections import OrderedDict
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.preprocessors.utils import get_feedforward_preprocessor
from tensorflow.python.keras.engine import training_utils
from softlearning.utils.keras import PicklableModel
import numpy as np


def get_rnd_networks_from_variant(variant, env):
    rnd_params = variant['algorithm_params']['rnd_params']
    target_network = None
    predictor_network = None

    observation_keys = variant['policy_params']['kwargs']['observation_keys']
    if not observation_keys:
        observation_keys = env.observation_keys
    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    inputs_flat = create_inputs(observation_shapes)

    target_network, predictor_network = [], []
    for input_tensor in inputs_flat:
        if 'pixels' in input_tensor.name: # check logic
            from softlearning.preprocessors.utils import get_convnet_preprocessor
            target_network.append(get_convnet_preprocessor(
                'rnd_target_conv',
                **rnd_params['convnet_params']
            )(input_tensor))
            predictor_network.append(get_convnet_preprocessor(
                'rnd_predictor_conv',
                **rnd_params['convnet_params']
            )(input_tensor))
        else:
            target_network.append(input_tensor)
            predictor_network.append(input_tensor)

    target_network = tf.keras.layers.Lambda(
        lambda inputs: tf.concat(training_utils.cast_if_floating_dtype(inputs), axis=-1)
    )(target_network)

    predictor_network = tf.keras.layers.Lambda(
        lambda inputs: tf.concat(training_utils.cast_if_floating_dtype(inputs), axis=-1)
    )(predictor_network)

    target_network = get_feedforward_preprocessor(
        'rnd_target_fc',
        **rnd_params['fc_params']
    )(target_network)

    predictor_network = get_feedforward_preprocessor(
        'rnd_predictor_fc',
        **rnd_params['fc_params']
    )(predictor_network)

    # Initialize RN weights
    target_network = PicklableModel(inputs_flat, target_network)
    target_network.set_weights([np.random.normal(0, 0.1, size=weight.shape)
                                for weight in target_network.get_weights()])
    predictor_network = PicklableModel(inputs_flat, predictor_network)
    return target_network, predictor_network
