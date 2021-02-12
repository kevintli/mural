import sys
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params, get_policy)
from softlearning.models.utils import (
    get_reward_classifier_from_variant, get_dynamics_model_from_variant)
from softlearning.misc.generate_goal_examples import (
    get_goal_example_from_variant, get_goal_transitions_from_variant)
# from softlearning.misc.get_multigoal_example_pools import (
#     get_example_pools_from_variant)
from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner


class ExperimentRunnerClassifierRL(ExperimentRunner):
    def _get_algorithm_kwargs(self, variant):
        algorithm_kwargs = super()._get_algorithm_kwargs(variant)
        algorithm_type = variant['algorithm_params']['type']

        # TODO: Replace this with a common API for single vs multigoal
        # === SINGLE GOAL POOL ===
        if algorithm_type in (
                'SACClassifier',
                'RAQ',
                'VICE',
                'VICEGAN',
                'VICERAQ',
                'VICEDynamicsAware',
                'DynamicsAwareEmbeddingVICE'):

            # from IPython import embed; embed()
            reward_classifier = self.reward_classifier = (
                get_reward_classifier_from_variant(
                    self._variant, algorithm_kwargs['training_environment']))
            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

            if algorithm_type == 'VICEDynamicsAware':
                algorithm_kwargs['dynamics_model'] = (get_dynamics_model_from_variant(
                    self._variant, algorithm_kwargs['training_environment']))

            elif algorithm_type == 'DynamicsAwareEmbeddingVICE':
                # TODO: Get this working for any environment
                self.distance_fn = algorithm_kwargs['distance_fn'] = (
                    reward_classifier.observations_preprocessors['state_observation'])
                # TODO: include goal state as one of the VICE goal exmaples? 
                algorithm_kwargs['goal_state'] = None

        # === LOAD GOAL POOLS FOR MULTI GOAL ===
        elif algorithm_type in (
                'VICEGANMultiGoal',
                'MultiVICEGAN'):
            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(variant))
            num_goals = len(goal_pools_train)

            reward_classifiers = self.reward_classifiers = tuple(
                get_reward_classifier_from_variant(
                    variant,
                    algorithm_kwargs['training_environment'])
                for _ in range(num_goals))

            algorithm_kwargs['classifiers'] = reward_classifiers
            algorithm_kwargs['goal_example_pools'] = goal_pools_train
            algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation

        elif algorithm_type == 'SQIL':
            goal_transitions = get_goal_transitions_from_variant(variant)
            algorithm_kwargs['goal_transitions'] = goal_transitions

        return algorithm_kwargs

    def _restore_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
        algorithm_kwargs = super()._restore_algorithm_kwargs(picklable, checkpoint_dir, variant)

        if 'reward_classifier' in picklable.keys():
            reward_classifier = self.reward_classifier = picklable[
                'reward_classifier']

            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

        if 'distance_estimator' in picklable.keys():
            distance_fn = self.distance_fn = picklable['distance_estimator']
            algorithm_kwargs['distance_fn'] = distance_fn
            algorithm_kwargs['goal_state'] = None

        return algorithm_kwargs

    def _restore_multi_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
        algorithm_kwargs = super()._restore_multi_algorithm_kwargs(
            picklable, checkpoint_dir, variant)

        if 'reward_classifiers' in picklable.keys():

            reward_classifiers = self.reward_classifiers = picklable[
                'reward_classifiers']
            for reward_classifier in self.reward_classifiers:
                reward_classifier.observation_keys = (variant['reward_classifier_params']
                                                             ['kwargs']
                                                             ['observation_keys'])

            algorithm_kwargs['classifiers'] = reward_classifiers
            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(variant))
            algorithm_kwargs['goal_example_pools'] = goal_pools_train
            algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation
        return algorithm_kwargs

    @property
    def picklables(self):
        picklables = super().picklables

        if hasattr(self, 'reward_classifier'):
            picklables['reward_classifier'] = self.reward_classifier
        elif hasattr(self, 'reward_classifiers'):
            picklables['reward_classifiers'] = self.reward_classifiers

        if hasattr(self, 'distance_fn'):
            picklables['distance_estimator'] = self.distance_fn

        return picklables


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `development.main`
    run_example_local('examples.classifier_rl', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
