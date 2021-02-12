
from collections import OrderedDict
from softlearning.models.convnet import convnet_model
import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.utils.keras import PicklableModel, PicklableSequential
from softlearning.preprocessors.utils import get_preprocessor_from_params
import numpy as np
from softlearning.environments.adapters.gym_adapter import GymAdapter
import gzip
import pickle
import glob
import os
import matplotlib.pyplot as plt

tfk = tf.keras
tfkl = tf.keras.layers

DEFAULT_STATE_ESTIMATOR_PREPROCESSOR_PARAMS = {
    'type': 'ConvnetPreprocessor',
    'kwargs': {
        'conv_filters': (64, ) * 4,
        'conv_kernel_sizes': (3, ) * 4,
        'conv_strides': (2, ) * 4,
        'normalization_type': None,
    },
}

def state_estimator_model(input_shape,
                          num_hidden_units=256,
                          num_hidden_layers=2,
                          output_size=4, # (x, y, z_cos, z_sin)
                          kernel_regularizer=None,
                          preprocessor_params=None,
                          preprocessor=None,
                          name='state_estimator_preprocessor'):
    # TODO: Make this take in observation keys instead of this hardcoded output size.
    obs_preprocessor_params = (
        preprocessor_params or DEFAULT_STATE_ESTIMATOR_PREPROCESSOR_PARAMS)
#     preprocessor = convnet_model(
#         name='convnet_preprocessor_state_est',
#         **convnet_kwargs)

    if preprocessor is None:
        preprocessor = get_preprocessor_from_params(None, obs_preprocessor_params)

    state_estimator = feedforward_model(
        hidden_layer_sizes=(num_hidden_units, ) * num_hidden_layers,
        output_size=output_size,
        output_activation=tf.keras.activations.tanh,
        kernel_regularizer=kernel_regularizer, # tf.keras.regularizers.l2(0.001),
        name='feedforward_state_est'
    )
    model = tfk.Sequential([
        tfk.Input(shape=input_shape,
                  name='pixels',
                  dtype=tf.uint8),
        preprocessor,
        state_estimator,
    ], name=name)
    return model


def get_dumped_pkl_data(pkl_path):
    with gzip.open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    assert 'pixels' in data and 'states' in data
    return data['pixels'], data['states']

def get_seed_data(seed_path):
    checkpoint_paths = [
        os.path.join(path, 'replay_pool.pkl')
        for path in sorted(glob.iglob(os.path.join(seed_path, 'checkpoint_*')))
    ]

    training_images = []
    ground_truth_states = []
    for checkpoint_path in checkpoint_paths:

        i = 0
        print(checkpoint_path)
        with gzip.open(checkpoint_path, 'rb') as f:
            pool = pickle.load(f)
            obs = pool['observations']
            training_images.append(obs['pixels'])
            pos = obs['object_position'][:, :2]
            pos = normalize(pos, -0.1, 0.1, -1, 1)
            num_samples = pos.shape[0]
            ground_truth_state = np.concatenate([
                pos,
                obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                obs['object_orientation_sin'][:, 2].reshape((num_samples, 1)),
            ], axis=1)
            ground_truth_states.append(ground_truth_state)

    return np.concatenate(training_images), np.concatenate(ground_truth_states)

def get_training_data(exp_path, limit=None):
    for exp in sorted(glob.iglob(os.path.join(exp_path, '*'))):
        training_images = None # np.array([])
        ground_truth_states = None # np.array([])

        if not os.path.isdir(exp):
            continue
        print(exp)
        checkpoint_paths = [
            os.path.join(path, 'replay_pool.pkl')
            for path in sorted(glob.iglob(os.path.join(exp, 'checkpoint_*')))
        ]

        training_images = []
        ground_truth_states = []
        for checkpoint_path in checkpoint_paths:
            i = 0
            print(checkpoint_path)
            with gzip.open(checkpoint_path, 'rb') as f:
                pool = pickle.load(f)
                obs = pool['observations']
                training_images.append(obs['pixels'])
                pos = obs['object_position'][:, :2]
                pos = normalize(pos, -0.1, 0.1, -1, 1)
                num_samples = pos.shape[0]
                ground_truth_state = np.concatenate([
                    pos,
                    obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                    obs['object_orientation_sin'][:, 2].reshape((num_samples, 1)),
                ], axis=1)
                ground_truth_states.append(ground_truth_state)

            i += 1
            if limit is not None and i == limit:
                break

    training_images = np.concatenate(training_images, axis=0)
    ground_truth_states = np.concatenate(ground_truth_states, axis=0)
    return training_images, ground_truth_states

def normalize(data, olow, ohigh, nlow, nhigh):
    """
    olow    old low
    ohigh   old high
    nlow    new low
    nhigh   new hight
    """
    percent = (data - olow) / (ohigh - olow)
    return percent * (nhigh - nlow) + nlow

def train(model, obs_keys_to_estimate, save_path, n_epochs=50):
    training_pools_base_path = '/root/softlearning-vice/goal_classifier/free_screw_state_estimator_data/all_data.pkl'

    if 'seed' in training_pools_base_path:
        pixels, states = get_seed_data(training_pools_base_path)
    elif 'pkl' in training_pools_base_path:
        pixels, states = get_dumped_pkl_data(training_pools_base_path)
    else:
        pixels, states = get_training_data(training_pools_base_path)

    history = model.fit(
        x=pixels,
        y=states,
        batch_size=64,
        epochs=n_epochs,
        validation_split=0.05,
    )

    model.save_weights(save_path)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./train_history.png')
    # plt.show()

    random_indices = np.random.choice(pixels.shape[0], size=50, replace=False)
    tests, labels = pixels[random_indices], states[random_indices]
    preds = model.predict(tests)

    pos_errors = []
    angle_errors = []
    import imageio
    for i, (test_img, label, pred) in enumerate(zip(tests, labels, preds)):
        pos_error_xy = np.abs(label[:2] - pred[:2])
        pos_error = np.sum(pos_error_xy)
        pos_error = 30 * pos_error # free box is 30 cm
        true_angle = np.arctan2(label[3], label[2])
        pred_angle = np.arctan2(pred[3], pred[2])
        angle_error = min(np.abs(true_angle - pred_angle), np.abs(pred_angle - true_angle))
        true_angle = true_angle * 180 / np.pi
        pred_angle = pred_angle * 180 / np.pi
        angle_error = angle_error * 180 / np.pi

        pos_errors.append(pos_error)
        angle_errors.append(angle_error)

        print('\n========== IMAGE #', i, '=========')
        print('POS ERROR (cm):', pos_error, 'true xy: {}'.format(label[:2]), 'pred xy: {}'.format(pred[:2]))
        print('ANGLE ERROR (degrees):', angle_error, 'true angle: {}'.format(true_angle), 'pred angle: {}'.format(pred_angle))
        imageio.imwrite(f'/root/imgs/test{i}.jpg', test_img)

    mean_pos_error = np.mean(pos_errors)
    mean_angle_error = np.mean(angle_errors)
    print('MEAN POS ERROR (CM):', mean_pos_error)
    print('MEAN ANGLE ERROR (degrees):', mean_angle_error)

if __name__ == '__main__':
    image_shape = (32, 32, 3)

    obs_keys = ('object_position',
                'object_orientation_cos',
                'object_orientation_sin')
    model = state_estimator_model(
        domain='DClaw',
        task='TurnFreeValve3ResetFreeSwapGoal-v0',
        obs_keys_to_estimate=obs_keys,
        input_shape=image_shape)

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='mean_squared_error')

    load_weights = False
    if load_weights:
        training_pools_base_path = '/root/softlearning-vice/goal_classifier/free_screw_state_estimator_data/all_data.pkl'
        weights_path = './state_estimator_random_data.h5'
        model.load_weights(weights_path)
        images, labels = get_dumped_pkl_data(training_pools_base_path)
        # images, labels = get_training_data(training_pools_base_path, limit=1)
        random_indices = np.random.randint(images.shape[0], size=250)
        tests = images[random_indices]
        preds = model.predict(tests)

        pos_errors = []
        angle_errors = []
        import imageio
        for i, (test_img, label, pred) in enumerate(zip(tests, labels, preds)):
            pos_error_xy = np.abs(label[:2] - pred[:2])
            pos_error = np.sum(pos_error_xy)
            pos_error = 30 * pos_error # free box is 30 cm
            true_angle = np.arctan2(label[3], label[2])
            pred_angle = np.arctan2(pred[3], pred[2])
            angle_error = min(np.abs(true_angle - pred_angle), np.abs(pred_angle - true_angle))
            true_angle = true_angle * 180 / np.pi
            pred_angle = pred_angle * 180 / np.pi
            angle_error = angle_error * 180 / np.pi

            pos_errors.append(pos_error)
            angle_errors.append(angle_error)

            print('\n========== IMAGE #', i, '=========')
            print('POS ERROR (cm):', pos_error, 'true xy: {}'.format(label[:2]), 'pred xy: {}'.format(pred[:2]))
            print('ANGLE ERROR (degrees):', angle_error, 'true angle: {}'.format(true_angle), 'pred angle: {}'.format(pred_angle))
            imageio.imwrite(f'/root/imgs/test{i}.jpg', test_img)

        ind = np.argpartition(pos_errors, -20)[-20:]
        ind = ind[np.argsort(pos_errors[ind])]
        top_error_imgs, top_error_labels, top_error_preds = images[ind], labels[ind], preds[ind]
        # for i, (img, label, pred) in enumerate(zip(top_error_imgs, top_error_labels, top_error_preds)):

        mean_pos_error = np.mean(pos_errors)
        mean_angle_error = np.mean(angle_errors)
        print('MEAN POS ERROR (CM):', mean_pos_error)
        print('MEAN ANGLE ERROR (degrees):', mean_angle_error)

    else:
        train(model, obs_keys, './state_estimator_invisible_claw.h5')
