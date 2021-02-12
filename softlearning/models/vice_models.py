import tensorflow as tf
from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_feedforward_reward_classifier_function(input_shapes,
                                                  *args,
                                                  preprocessors=None,
                                                  observation_keys=None,
                                                  name='feedforward_reward_classifier',
                                                  kernel_regularizer_lambda=1e-3,
                                                  # output_activation=tf.math.log_sigmoid,
                                                  **kwargs):
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    reward_classifier_function = feedforward_model(
        *args,
        output_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer_lambda) if kernel_regularizer_lambda else None,
        name=name,
        # output_activation=output_activation,
        **kwargs)

    # from IPython import embed; embed()
    reward_classifier_function = PicklableModel(
        inputs_flat, reward_classifier_function(preprocessed_inputs))
    reward_classifier_function.observation_keys = observation_keys
    reward_classifier_function.observations_preprocessors = preprocessors

    return reward_classifier_function

