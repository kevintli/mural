from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_embedding_fn(input_shapes,
                        embedding_dim,
                        *args,
                        preprocessors=None,
                        observation_keys=None,
                        goal_keys=None,
                        name='embedding_fn',
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

    embedding_fn = feedforward_model(
        *args, output_size=embedding_dim, name=f'feedforward_{name}', **kwargs)

    embedding_fn = PicklableModel(
        inputs_flat, embedding_fn(preprocessed_inputs),
        name=name)

    embedding_fn.observation_keys = observation_keys or tuple()
    embedding_fn.goal_keys = goal_keys or tuple()
    embedding_fn.all_keys = embedding_fn.observation_keys + embedding_fn.goal_keys

    return embedding_fn

def create_distance_estimator(input_shapes,
                              *args,
                              preprocessors=None,
                              observation_keys=None,
                              goal_keys=None,
                              name='distance_estimator',
                              classifier_params=None,
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

    output_size = 1 if not classifier_params else int(classifier_params.get('bins', 1) + 1)

    distance_fn = feedforward_model(
        *args,
        output_size=output_size,
        name=name,
        **kwargs)

    distance_fn = PicklableModel(inputs_flat, distance_fn(preprocessed_inputs))
    # preprocessed_inputs_fn = PicklableModel(inputs_flat, preprocessed_inputs)

    distance_fn.observation_keys = observation_keys or tuple()
    distance_fn.goal_keys = goal_keys or tuple()
    distance_fn.all_keys = distance_fn.observation_keys + distance_fn.goal_keys
    distance_fn.classifier_params = classifier_params

    # distance_fn.observations_preprocessors = preprocessors['s1']
    # distance_fn.preprocessed_inputs_fn = preprocessed_inputs_fn
    return distance_fn
