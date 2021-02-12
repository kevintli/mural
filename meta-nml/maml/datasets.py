import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid, ModelMLPToy2D
from maml.utils import ToTensor1D
from maml.metadatasets.nml import NMLToy2D, NoisyDuplicatesProblem

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    """
    Returns a namedtuple with the train/val/test split, model, and loss function
    for the specified task.

    Parameters
    ----------
    name : str
        Name of the dataset to use

    folder : str
        Folder where dataset is stored (or will download to this path if not found)

    num_ways : int
        Number of classes for each task
    
    num_shots : int
        Number of training examples provided per class

    num_shots_test : int
        Number of test examples provided per class (during adaptation)
    """
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'nmltoy2d':
        model_hidden_sizes=[1024, 1024]
        replay_pool_size = 100
        clip_length = 100
        from_beginning = False

        # For validation and testing, we evaluate the outer loss on the entire dataset;
        # for testing, we use smaller batches for efficiency
        meta_train_dataset = NMLToy2D(replay_pool_size=replay_pool_size, 
                                      clip_length=clip_length,
                                      from_beginning=from_beginning,
                                      test_strategy='sample',
                                      test_batch_size=10)
        meta_val_dataset = NMLToy2D(replay_pool_size=replay_pool_size, 
                                      clip_length=clip_length,
                                      from_beginning=from_beginning,
                                      test_strategy='all')
        meta_test_dataset = NMLToy2D(replay_pool_size=replay_pool_size, 
                                      clip_length=clip_length,
                                      from_beginning=from_beginning,
                                      test_strategy='all')

        model = ModelMLPToy2D(model_hidden_sizes)
        loss_function = F.cross_entropy

    elif name == 'noisyduplicates':
        model_hidden_sizes=[2048, 2048]
        locations = [
            ([-2.5, 2.5], 1, 0), # Single visit (negative)
            ([2.5, 2.5], 10, 0), # Many visits
            ([-2.5, -2.5], 2, 15), # A few negatives, mostly positives
            ([2.5, -2.5], 8, 15) # More negatives, but still majority positives
        ]
        noise_std = 0

        meta_train_dataset = NoisyDuplicatesProblem(locations, noise_std=noise_std)
        meta_val_dataset = NoisyDuplicatesProblem(locations, noise_std=noise_std)
        meta_test_dataset = NoisyDuplicatesProblem(locations, noise_std=noise_std)

        model = ModelMLPToy2D(model_hidden_sizes)
        loss_function = F.cross_entropy

    elif name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
