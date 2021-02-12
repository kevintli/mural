from copy import deepcopy


def create_SAC_Classifier_algorithm(variant, *args, **kwargs):
    from .sac_classifier import SACClassifier
    algorithm = SACClassifier(*args, **kwargs)
    return algorithm


def create_RAQ_algorithm(variant, *args, **kwargs):
    from .raq import RAQ
    algorithm = RAQ(*args, **kwargs)
    return algorithm


def create_VICE_algorithm(variant, *args, **kwargs):
    from .vice import VICE
    algorithm = VICE(*args, **kwargs)
    return algorithm


def create_VICEDynamicsAware_algorithm(variant, *args, **kwargs):
    from .vice_dynamics_aware import VICEDynamicsAware
    algorithm = VICEDynamicsAware(*args, **kwargs)
    return algorithm


def create_VICEGANTwoGoal_algorithm(variant, *args, **kwargs):
    from .vice_multigoal import VICEGANTwoGoal
    algorithm = VICEGANTwoGoal(*args, **kwargs)
    return algorithm


def create_VICEGANMultiGoal_algorithm(variant, *args, **kwargs):
    from .vice_multigoal import VICEGANMultiGoal
    algorithm = VICEGANMultiGoal(*args, **kwargs)
    return algorithm


def create_VICE_GAN_algorithm(variant, *args, **kwargs):
    from .vice_gan import VICEGAN
    algorithm = VICEGAN(*args, **kwargs)
    return algorithm


def create_VICE_RAQ_algorithm(variant, *args, **kwargs):
    from .viceraq import VICERAQ
    algorithm = VICERAQ(*args, **kwargs)
    return algorithm

def create_VICEGoalConditioned_algorithm(variant, *args, **kwargs):
    from .vice_goal_conditioned import VICEGoalConditioned
    algorithm = VICEGoalConditioned(*args, **kwargs)
    return algorithm

def create_VICEGANGoalConditioned_algorithm(variant, *args, **kwargs):
    from .vice_gan_goal_conditioned import VICEGANGoalConditioned
    algorithm = VICEGANGoalConditioned(*args, **kwargs)
    return algorithm


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC
    algorithm = SAC(*args, **kwargs)
    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL
    algorithm = SQL(*args, **kwargs)
    return algorithm


def create_MultiSAC_algorithm(variant, *args, **kwargs):
    from .multi_sac import MultiSAC
    algorithm = MultiSAC(*args, **kwargs)
    return algorithm


def create_MultiVICEGAN_algorithm(variant, *args, **kwargs):
    from .multi_vice import MultiVICEGAN
    algorithm = MultiVICEGAN(*args, **kwargs)
    return algorithm


def create_SQIL_algorithm(variant, *args, **kwargs):
    from .sqil import SQIL
    algorithm = SQIL(*args, **kwargs)
    return algorithm


def create_DDL_algorithm(variant, *args, **kwargs):
    from .ddl import DDL
    algorithm = DDL(*args, **kwargs)
    return algorithm


def create_DynamicsAwareEmbeddingDDL_algorithm(variant, *args, **kwargs):
    from .ddl import DynamicsAwareEmbeddingDDL
    algorithm = DynamicsAwareEmbeddingDDL(*args, **kwargs)
    return algorithm


def create_DynamicsAwareEmbeddingVICE_algorithm(variant, *args, **kwargs):
    from .ddl import DynamicsAwareEmbeddingVICE
    algorithm = DynamicsAwareEmbeddingVICE(*args, **kwargs)
    return algorithm


def create_HERQLearning_algorithm(variant, *args, **kwargs):
    from .her_qlearning import HERQLearning
    algorithm = HERQLearning(*args, **kwargs)
    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'MultiSAC': create_MultiSAC_algorithm,
    'SQL': create_SQL_algorithm,

    # === Classifier Based Methods ===
    'SACClassifier': create_SAC_Classifier_algorithm,
    'RAQ': create_RAQ_algorithm,
    'VICE': create_VICE_algorithm,
    'VICEDynamicsAware': create_VICEDynamicsAware_algorithm,
    'VICEGAN': create_VICE_GAN_algorithm,
    'VICERAQ': create_VICE_RAQ_algorithm,
    'VICEGoalConditioned': create_VICEGoalConditioned_algorithm,
    'VICEGANGoalConditioned': create_VICEGANGoalConditioned_algorithm,
    'VICEGANTwoGoal': create_VICEGANTwoGoal_algorithm,
    'VICEGANMultiGoal': create_VICEGANMultiGoal_algorithm,
    'MultiVICEGAN': create_MultiVICEGAN_algorithm,

    # === Hindsight Methods ===
    'SQIL': create_SQIL_algorithm,
    'HERQLearning': create_HERQLearning_algorithm,

    # === Dynamical Distance Methods ===
    'DDL': create_DDL_algorithm,
    'DynamicsAwareEmbeddingDDL': create_DynamicsAwareEmbeddingDDL_algorithm,
    'DynamicsAwareEmbeddingVICE': create_DynamicsAwareEmbeddingVICE_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']

    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
