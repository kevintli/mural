import os
import copy
import glob
import pickle
import sys
import numpy as np

import tensorflow as tf
from ray import tune

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params)
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local

from softlearning.samplers.nn_sampler import NNSampler
import numpy as np

tf.compat.v1.disable_eager_execution()
from softlearning.rnd.utils import get_rnd_networks_from_variant
from pathlib import Path


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self._session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(self._session)

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()

    def _get_algorithm_kwargs(self, variant):
        self._multi_build = (variant['algorithm_params']['type'] in ['MultiSAC', 'MultiVICEGAN'])
        if self._multi_build:
            return self._get_multi_algorithm_kwargs(variant)

        environment_params = variant['environment_params']
        # from IPython import embed; embed(header='env params')
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else copy.deepcopy(training_environment))

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        policy = self.policy = get_policy_from_variant(
            variant, training_environment)

        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        Q_targets = self.Q_targets = get_Q_function_from_variant(
            variant, training_environment)

        # Set the preprocessors to be the same
        preprocessor_params = variant['Q_params']['kwargs']['observation_preprocessors_params']
        for observation_name, params in preprocessor_params.items():
            if params and params.get('shared', False):
                print(f'Setting *{observation_name}* preprocessors to be the same across policy/Qs')
                preprocessor = Qs[0].observations_preprocessors[observation_name]
                policy.preprocessors[observation_name] = preprocessor
                assert (
                    policy.preprocessors[observation_name] is preprocessor and
                    Qs[1].observations_preprocessors[observation_name] is preprocessor)

        # ==== LOADING IN CONVNET EXPERIMENT ====
        if ('pixels' in preprocessor_params
                and 'ConvnetPreprocessor' == preprocessor_params['pixels']['type']
                and preprocessor_params['pixels'].get('weights_path', None) is not None):
            weights_path = preprocessor_params['pixels']['weights_path']
            with open(weights_path, 'rb') as f:
                weights = pickle.load(f)

                def set_weights_and_fix(model):
                    model.set_weights(weights)
                    model.trainable = False

                set_weights_and_fix(self.policy.preprocessors['pixels'])
                set_weights_and_fix(self.Qs[0].observations_preprocessors['pixels'])
                set_weights_and_fix(self.Qs[1].observations_preprocessors['pixels'])
                set_weights_and_fix(self.Q_targets[0].observations_preprocessors['pixels'])
                set_weights_and_fix(self.Q_targets[1].observations_preprocessors['pixels'])

        # # === LOGGING STATE ESTIMATOR PREPROCESSOR OUTPUTS ===
        # if (self.policy.preprocessors.get('pixels', None)
        #     and self.policy.preprocessors['pixels'].__class__.__name__ == 'state_estimator_preprocessor'):
        #     state_estimator = self.policy.preprocessors['pixels']

        #     from softlearning.replay_pools.flexible_replay_pool import Field
        #     replay_pool = self.replay_pool = (
        #         get_replay_pool_from_variant(variant, training_environment,
        #             extra_obs_keys_and_fields={
        #                 'object_state_prediction': Field(
        #                     name='object_state_prediction',
        #                     dtype=np.float32,
        #                     shape=(4,)
        #                 )
        #             }))
        # else:
        #     state_estimator = None

        sampler = self.sampler = get_sampler_from_variant(variant)
            # state_estimator=state_estimator)

        last_checkpoint_dir = variant['replay_pool_params'].get(
            'last_checkpoint_dir', None)
        if last_checkpoint_dir:
            print('restoring')
            self._restore_replay_pool(last_checkpoint_dir)

        if isinstance(sampler, NNSampler):
            print('restoring nn_pool')
            nn_pool_dir = variant['sampler_params']['nn_pool_dir']
            nn_pool = (get_replay_pool_from_variant(variant, training_environment))

            replay_pool = self.replay_pool
            self.replay_pool = nn_pool
            self._restore_replay_pool(nn_pool_dir)
            self.replay_pool = replay_pool
            self.sampler.initialize_nn_pool(nn_pool)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'], training_environment))

        # === INITIALIZE RND NETWORKS ===
        self.rnd_networks = (
            get_rnd_networks_from_variant(variant, training_environment)
            if 'rnd_params' in variant['algorithm_params']
            else ()
        )

        # === PASS VAE INTO ALGORITHM TO GET DIAGNOSTICS ===
        # using_vae = (
        #     'pixels' in self.policy.preprocessors
        #     and self.policy.preprocessors['pixels'].name == 'vae_preprocessor')
        # from softlearning.models.utils import get_vae
        # vae = (
        #     get_vae(**variant['policy_params']
        #                      ['kwargs']
        #                      ['observation_preprocessors_params']
        #                      ['pixels']
        #                      ['kwargs'])
        #     if using_vae else None
        # )

        algorithm_kwargs = {
            'variant': variant,
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'Q_targets': Q_targets,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
            'rnd_networks': self.rnd_networks,
            # 'vae': vae,
            # 'state_estimator': state_estimator,
        }
        return algorithm_kwargs

    def _get_multi_algorithm_kwargs(self, variant):
        self._share_pool = variant['algorithm_params']['kwargs'].pop('share_pool')

        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else copy.deepcopy(training_environment))
        num_goals = training_environment.num_goals

        if self._share_pool:
            self.replay_pool = get_replay_pool_from_variant(variant, training_environment)
            replay_pools = self._replay_pools = tuple([
                self.replay_pool for _ in range(num_goals)
            ])
        else:
            replay_pools = self._replay_pools = tuple([
                get_replay_pool_from_variant(variant, training_environment)
                for _ in range(num_goals)
            ])

        samplers = self._samplers = tuple([
            get_sampler_from_variant(variant)
            for _ in range(num_goals)
        ])

        Qs_per_policy = self._Qs_per_policy = tuple([
            get_Q_function_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])
        Q_targets_per_policy = self._Qs_per_policy = tuple([
            get_Q_function_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])

        policies = self._policies = tuple([
            get_policy_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])

        # Set the preprocessors to be the same thing for each (Qs, policy) pair
        preprocessor_params = variant['Q_params']['kwargs']['observation_preprocessors_params']
        for observation_name, params in preprocessor_params.items():
            if params and params.get('shared', False):
                print(f'Setting *{observation_name}* preprocessors to be the same across policy/Qs')
                for Qs, policy in zip(Qs_per_policy, policies):
                    preprocessor = Qs[0].observations_preprocessors[observation_name]
                    policy.preprocessors[observation_name] = preprocessor
                    assert (
                        policy.preprocessors[observation_name] is preprocessor and
                        Qs[1].observations_preprocessors[observation_name] is preprocessor)


        if 'last_checkpoint_dir' in variant['replay_pool_params']:
            last_checkpoint_dir = variant['replay_pool_params']['last_checkpoint_dir']
        else:
            last_checkpoint_dir = None

        if last_checkpoint_dir:
            print('restoring')
            self._restore_replay_pool(last_checkpoint_dir)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'], training_environment))

        from softlearning.rnd.utils import get_rnd_networks_from_variant
        self.rnd_networks = (
            [get_rnd_networks_from_variant(variant, training_environment)
                for _ in range(num_goals)]
            if variant['algorithm_params'].get('rnd_params', None)
            else ()
        )

        algorithm_kwargs = {
            'variant': variant,
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policies': policies,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs_per_policy': Qs_per_policy,
            'Q_targets_per_policy': Q_targets_per_policy,
            'pools': replay_pools,
            'samplers': samplers,
            'num_goals': num_goals,
            'rnd_networks': self.rnd_networks,
            'session': self._session
        }
        return algorithm_kwargs

    def _build(self):
        variant = copy.deepcopy(self._variant)

        # from IPython import embed; embed()

        algorithm_kwargs = self._get_algorithm_kwargs(variant)

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _policy_params_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy_params.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _replay_pools_pickle_paths(self, checkpoint_dir):
        return [os.path.join(checkpoint_dir, f'replay_pool_{i}.pkl')
                for i in range(len(self._replay_pools))]

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    @property
    def picklables(self):
        # TODO: Save the VAE preprocessor weights, load them back in on restoration
        if self._multi_build:
            return {
                'variant': self._variant,
                'training_environment': self.training_environment,
                'evaluation_environment': self.evaluation_environment,
                # 'samplers': self._samplers, don't save sampler. Involves saving entire pool.
                'algorithm': self.algorithm,
                'policy_weights': [policy.get_weights() for policy in self._policies],
                'rnd_networks': self.rnd_networks,
            }
        return {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            # 'sampler': self.sampler,
            'algorithm': self.algorithm,
            'policy_weights': self.policy.get_weights(),
            'rnd_networks': self.rnd_networks,
        }

    def _save_value_functions(self, checkpoint_dir):
        if self._multi_build:
            for i, Qs in enumerate(self._Qs_per_policy):
                for j, Q in enumerate(Qs):
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'Qs_{i}_{j}')
                    Q.save_weights(checkpoint_path)
                if 'pixels' in Q.observations_preprocessors:
                    preprocessor_path = os.path.join(
                        checkpoint_dir,
                        f'Q_pixels_preprocessor_config_{i}.pkl')
                    with open(preprocessor_path, 'wb') as f:
                        pickle.dump(Q.observations_preprocessors['pixels'].get_config(), f)
        else:
            if isinstance(self.Qs, tf.keras.Model):
                Qs = [self.Qs]
            elif isinstance(self.Qs, (list, tuple)):
                Qs = self.Qs
            else:
                raise TypeError(self.Qs)
            for i, Q in enumerate(Qs):
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'Qs_{i}')
                Q.save_weights(checkpoint_path)
            if 'pixels' in Q.observations_preprocessors:
                preprocessor_path = os.path.join(
                    checkpoint_dir,
                    f'Q_pixels_preprocessor_config.pkl')
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(Q.observations_preprocessors['pixels'].get_config(), f)


    def _restore_value_functions(self, checkpoint_dir):
        if self._multi_build:
            for i, Qs in enumerate(self._Qs_per_policy):
                for j, Q in enumerate(Qs):
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'Qs_{i}_{j}')
                    Q.load_weights(checkpoint_path)
            for i, Q_targets in enumerate(self._Q_targets_per_policy):
                for j, Q in enumerate(Q_targets):
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'Qs_{i}_{j}')
                    Q.load_weights(checkpoint_path)

        else:
            if isinstance(self.Qs, tf.keras.Model):
                Qs = [self.Qs]
            elif isinstance(self.Qs, (list, tuple)):
                Qs = self.Qs
            else:
                raise TypeError(self.Qs)

            for i, Q in enumerate(Qs):
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'Qs_{i}')
                Q.load_weights(checkpoint_path)

    def _save_rnd_networks(self, checkpoint_dir):
        if self._multi_build:
            for i, rnd_network_pair in enumerate(self.rnd_networks):
                target_network, predictor_network = self.rnd_networks[i]
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'rnd_target_{i}')
                target_network.save_weights(checkpoint_path)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'rnd_predictor_{i}')
                predictor_network.save_weights(checkpoint_path)
        else:
            target_network, predictor_network = self.rnd_networks
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'rnd_target')
            target_network.save_weights(checkpoint_path)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'rnd_predictor')
            predictor_network.save_weights(checkpoint_path)

    def _restore_rnd_networks(self, checkpoint_dir):
        if self._multi_build:
            for i, rnd_network_pair in enumerate(self.rnd_networks):
                target_network, predictor_network = self.rnd_networks[i]
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'rnd_target_{i}')
                target_network.load_weights(checkpoint_path)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'rnd_predictor_{i}')
                predictor_network.load_weights(checkpoint_path)
        else:
            target_network, predictor_network = self.rnd_networks
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'rnd_target')
            target_network.load_weights(checkpoint_path)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'rnd_predictor')
            predictor_network.load_weights(checkpoint_path)

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO: This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            try:
                self.evaluation_environment._env.grid_render = None
            except Exception:
                pass
            pickle.dump(self.picklables, f)

        policy_params_path = self._policy_params_path(checkpoint_dir)
        with open(policy_params_path, 'wb') as f:
            pickle.dump(self.picklables['policy_weights'], f)

        self._save_value_functions(checkpoint_dir)
        if self.rnd_networks:
            self._save_rnd_networks(checkpoint_dir)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            if self._multi_build:
                self._save_replay_pools(checkpoint_dir)
            else:
                self._save_replay_pool(checkpoint_dir)

        tf_checkpoint = self._get_tf_checkpoint()

        tf_checkpoint.save(
            file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),
            session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _save_replay_pools(self, checkpoint_dir):
        replay_pools_pickle_paths = self._replay_pools_pickle_paths(
            checkpoint_dir)

        if self._share_pool:
            self._save_replay_pool(checkpoint_dir)
        else:
            for i, replay_pool in enumerate(self._replay_pools):
                self._replay_pools[i].save_latest_experience(
                    replay_pools_pickle_paths[i])

    def _restore_replay_pool(self, current_checkpoint_dir):
        # experiment_root = os.path.dirname(current_checkpoint_dir)
        experiment_root = Path(current_checkpoint_dir).parent
        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _restore_replay_pools(self, current_checkpoint_dir):
        experiment_root = Path(current_checkpoint_dir).parent

        experience_paths_per_replay_pool = [
            self._replay_pools_pickle_paths(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_paths in experience_paths_per_replay_pool:
            for i, experience_path in enumerate(experience_paths):
                self._replay_pools[i].load_experience(experience_path)

    def _restore_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))

        if variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = get_sampler_from_variant(
            variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        self._restore_value_functions(checkpoint_dir)
        policy = self.policy = (
            get_policy_from_variant(variant, training_environment))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'],
                training_environment))

        if variant['algorithm_params']['rnd_params']:
            from softlearning.rnd.utils import get_rnd_networks_from_variant
            self.rnd_networks = get_rnd_networks_from_variant(variant, training_environment)
            self._restore_rnd_networks(checkpoint_dir)
        else:
            self.rnd_networks = ()

        algorithm_kwargs = {
            'variant': variant,
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
            'rnd_networks': self.rnd_networks
        }
        return algorithm_kwargs

    def _restore_multi_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
        self._share_pool = variant['algorithm_params']['kwargs'].pop('share_pool')

        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        num_goals = training_environment.num_goals

        if self._share_pool:
            self.replay_pool = get_replay_pool_from_variant(variant, training_environment)
            self._restore_replay_pool(checkpoint_dir)
            replay_pools = self._replay_pools = tuple([
                self.replay_pool for _ in range(num_goals)
            ])
        else:
            replay_pools = self._replay_pools = tuple([
                get_replay_pool_from_variant(variant, training_environment)
                for _ in range(num_goals)
            ])
            self._restore_replay_pools(checkpoint_dir)

        samplers = self._samplers = tuple([
            get_sampler_from_variant(variant)
            for _ in range(num_goals)
        ])

        Qs_per_policy = self._Qs_per_policy = tuple([
            get_Q_function_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])
        Q_targets_per_policy = self._Q_targets_per_policy = tuple([
            get_Q_function_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])

        self._restore_value_functions(checkpoint_dir)

        policies = self._policies = tuple([
            get_policy_from_variant(variant, training_environment)
            for _ in range(num_goals)
        ])
        for policy, policy_weights in zip(self._policies, picklable['policy_weights']):
            policy.set_weights(policy_weights)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'],
                training_environment))

        if variant['algorithm_params']['rnd_params']:
            from softlearning.rnd.utils import get_rnd_networks_from_variant
            self.rnd_networks = [get_rnd_networks_from_variant(variant, training_environment) for _ in range(num_goals)]
            self._restore_rnd_networks(checkpoint_dir)
        else:
            self.rnd_networks = ()

        algorithm_kwargs = {
            'variant': variant,
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policies': policies,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs_per_policy': Qs_per_policy,
            'Q_targets_per_policy': Q_targets_per_policy,
            'pools': replay_pools,
            'samplers': samplers,
            'session': self._session,
            'rnd_networks': self.rnd_networks,
            'num_goals': num_goals,
        }
        return algorithm_kwargs

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        variant = copy.deepcopy(self._variant)

        self._multi_build = (self._variant['algorithm_params']['type'] in ['MultiSAC', 'MultiVICEGAN'])

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        if self._multi_build:
            algorithm_kwargs = self._restore_multi_algorithm_kwargs(picklable, checkpoint_dir, variant)
        else:
            algorithm_kwargs = self._restore_algorithm_kwargs(picklable, checkpoint_dir, variant)

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO: target Qs should either be checkpointed or pickled.
        if self._multi_build:
            for Qs, Q_targets in zip(self.algorithm._Qs_per_policy, self.algorithm._Q_targets_per_policy):
                for Q, Q_target in zip(Qs, Q_targets):
                    Q_target.set_weights(Q.get_weights())
        else:
            for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
                Q_target.set_weights(Q.get_weights())

        self._built = True
        print("Finished Restoring")


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.development', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
