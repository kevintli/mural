from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_dynamics_model(input_shapes,
                          dynamics_latent_dim,
                          *args,
                          preprocessors=None,
                          observation_keys=None,
                          goal_keys=None,
                          name='dynamics_model',
                          encoder_kwargs=None,
                          decoder_kwargs=None,
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
    encoder = feedforward_model(
        *args,
        output_size=dynamics_latent_dim,
        name=f'{name}_encoder',
        **encoder_kwargs)

    output_size = sum([
        shape.as_list()[0]
        for shape in input_shapes['observations'].values()
    ])
    decoder = feedforward_model(
        *args,
        output_size=output_size,
        name=f'{name}_decoder',
        **decoder_kwargs)

    latent = encoder(preprocessed_inputs)
    dynamics_pred = decoder(latent)

    dynamics_model = PicklableModel(inputs_flat, dynamics_pred, name=name)

    dynamics_model.observation_keys = observation_keys or tuple()
    dynamics_model.goal_keys = goal_keys or tuple()
    dynamics_model.all_keys = dynamics_model.observation_keys + dynamics_model.goal_keys

    dynamics_model.encoder = PicklableModel(inputs_flat, latent, name=f'{name}_encoder_model')

    return dynamics_model
