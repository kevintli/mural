from collections import OrderedDict
from math import ceil
import os

import matplotlib.pyplot as plt
import numpy as np

from torchmeta.utils.data import Task, MetaDataset
from maml.tasks import TensorTask
from maml.utils import ToTensor1D, load_tfds_as_torchvision

import torch
import torchvision
from torchvision import transforms

def get_shuffled(*args):
    idxs = np.random.permutation(range(len(args[0])))
    results = []
    for tensor in args:
        results.append(tensor[idxs])
    return results if len(results) > 1 else results[0]

class NML(MetaDataset):
    """
    Dataset for an NML meta-learning task. The goal is to learn a representation that can be easily 
    finetuned to a query point with an arbitrary label by taking gradient steps on *just that query point*.

    The dataset is constructed by converting each input from train_points into 
    `num_classes` tasks (one for each proposed label).

    Parameters
    ----------
    train_data : tuple[np.array, np.array]
        Inputs and labels to sample from for adaptation (inner loop).
        Each task will consist of one of these inputs with a proposed class label.
    
    test_data : tuple[np.array, np.array]
        Inputs and labels to use for the test loss (outer loop).

    points_per_task : int, optional
        Number of points to use for a single task, including the query point itself.
        If this is >1, then points other than the query point will be used for adaptation as well.

    test_strategy: str in {'all', 'sample', 'cycle'}
        The strategy to use for choosing inputs/labels for evaluating the test loss
        (outer loss in MAML). 

        'all': use entire test set for each task
        'sample': sample batches each time
        'cycle': cycle through batches from the test set, so that each task when evaluated
                 for the i-th time will use the i-th test batch

    test_batch_size: int, optional
        The batch size to use for test loss. Only used if the test_strategy is 'sample' or 'cycle'.

    include_query_point : bool, optional
        Whether to include the downweighted query point in every batch during testing. Only used
        if the test_strategy is 'sample' or 'cycle'. Default: True

    dist_weight_thresh : float, optional
        If provided, will weight points in the outer loss according to L2 distance from the query point
        (with farther points receiving lower weight). `dist_weight_thresh` is the distance at which the weight becomes 0.1

        Specifically, the formula for the weight on a point x given query point q is:
            exp(-||x - q|| * 2.3 / dist_weight_thresh)

    equal_pos_neg_test : bool, optional
        For VICE setup only. If True, will balance the number of positive and negative examples
        sampled for the test set.

    query_point_weight: int, optional
        The weight to place on the query point in the test loss. Default: 1

    num_classes : int
        Number of classes to use for NML. Default: 2

    num_tasks : int, optional
        Number of tasks to create from the dataset. By default, we create num_classes tasks per point
         (one for each proposed label), resulting in num_classes x (# of dataset points) tasks.

    concat_query_point : bool, optional
        Whether to concatenate the query point to each input to feed into the model during adaptation/evaluation.
        If True, then for a d-dimensional dataset, each input x will be 2d-dimensional
        and have the form [x, query_point]

    mixup_alpha : float, optional
        Alpha parameter to use for mixup (https://arxiv.org/pdf/1710.09412.pdf)
        (only affects the test set, not the tasks themselves).

    transform : object
        Torch transform for inputs

    target_transform : object
        Torch transform for labels

    dataset_transform : object
        Torch transform for Task objects
    """
    def __init__(self, train_data, test_data=None, points_per_task=1, test_strategy='all', test_batch_size=0, 
                include_query_point=True, dist_weight_thresh=None, inner_loop_dist_weight=True,
                equal_pos_neg_test=False, query_point_weight=1, kernel=None,
                concat_query_point=False, mixup_alpha=None, num_classes=2, num_tasks=0, transform=None, 
                target_transform=None, dataset_transform=None):
        print(self.__class__)
        print(isinstance(self, NML))
        
        super(NML, self).__init__(meta_split='train', 
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.train_data = train_data
        self._train_inputs, self._train_labels, self._train_features = get_shuffled(*self.train_data)
        self.test_data = test_data or train_data
        self._test_inputs, self._test_labels, self._test_features = get_shuffled(*self.test_data)

        self.points_per_task = points_per_task

        assert test_strategy in ['all', 'sample', 'cycle'], \
            f"Invalid test_strategy: '{test_strategy}'. Must choose from ['all', 'sample', 'cycle']."
        self.test_strategy = test_strategy
        self.include_query_point = include_query_point
        self.dist_weight_thresh = dist_weight_thresh
        self.inner_loop_dist_weight = inner_loop_dist_weight
        self.equal_pos_neg_test = equal_pos_neg_test
        self.query_point_weight = query_point_weight
        self.kernel = kernel

        self._cycle_indices = {}
        if self.include_query_point:
            # Include the query point as the last item in every batch (downweighted appropriately).
            self.test_batch_size = (test_batch_size - 1) if test_batch_size else len(self._test_inputs)
            self._num_batches = ceil(len(self._test_inputs) / self.test_batch_size)
        else:
            # Consider the query point as part of the test set, so that it can be sampled
            # just like the others (but not necessarily in every batch).
            self.test_batch_size = test_batch_size or len(self._test_inputs) + 1
            self._num_batches = ceil((len(self._test_inputs) + 1) / self.test_batch_size)

        self.concat_query_point = concat_query_point
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.transform = transform

        assert num_tasks <= self.num_classes * len(self._train_inputs), "Cannot have more tasks than (num classes) x (total training points)"
        self.num_tasks = num_tasks or self.num_classes * len(self._train_inputs)

    def _get_train_task(self, index, point_index, proposed_label):
        """
        Constructs a training task from a single query point and proposed label.
        """
        query_point = self._train_inputs[point_index]
        query_feat = self._train_features[point_index]
        X, y, X_feats = query_point[None], np.array([proposed_label]), query_feat[None]

        if self.points_per_task > 1:
            # Sample points_per_task-1 additional points for the task, but NOT the query point again
            idxs = np.random.permutation(range(len(self._test_inputs)))[:self.points_per_task-1]
            X = np.vstack((X, self._test_inputs[idxs]))
            y = np.hstack((y, self._test_labels[idxs])).astype(np.long)
            X_feats = np.vstack((X_feats, self._test_features[idxs]))
            if self.dist_weight_thresh and self.inner_loop_dist_weight:
                # Weight points exponentially lower as distance from the query point increases.
                # (since the goal is just to have good finetuning accuracy on the query point)
                weights = np.exp(-np.linalg.norm(query_feat - X_feats, axis=-1) * 2.3 / self.dist_weight_thresh)
                # weights = self.kernel.embed(query_point, X) #np.exp(-np.linalg.norm(self.weight_embedding(query_point) - self.weight_embedding(X), axis=-1) * 2.3 / self.dist_weight_thresh)
            else:
                weights = np.ones(len(X))

            if self.include_query_point and self.test_strategy in ['sample', 'cycle']:
                num_batches = ceil(len(self._test_inputs) / self.points_per_task)
                weights[0] = weights[0] / num_batches * self.query_point_weight

            y = np.hstack((y[:,None], weights[:,None]))

        return self._to_tensor_task(index, X, y)
    
    
    def _get_test_task(self, index, point_index, proposed_label):
        """
        Constructs a test set for a given task (specified by `point_index` and `proposed_label`),
        consisting of points in the dataset and possibly the query point.
        """
        query_point = self._train_inputs[point_index]
        query_feat = self._train_features[point_index]

        ## Construct the set of potential test points to sample from
        if self.test_strategy in ['sample', 'cycle'] and self.include_query_point:
            # Don't add query point to the sampling pool; it will automatically be added
            # to every batch and downweighted appropriately.
            all_inputs, all_labels, all_feats = self._test_inputs, self._test_labels, self._test_features
        else:
            # Treat the query point as just another point that can be sampled
            all_inputs = np.vstack([self._test_inputs, query_point])
            all_labels = np.hstack([self._test_labels, proposed_label])
            all_feats = np.vstack([self._test_features, query_feat])

        ## Sample a batch according to `test_strategy`
        if self.test_strategy == 'all':
            X, y, X_feats = all_inputs, all_labels, all_feats
        elif self.test_strategy == 'sample':
            if self.equal_pos_neg_test:
                # Sample an equal number of positive and negative examples
                pos_inputs = all_inputs[all_labels == 1]
                pos_feats = all_feats[all_labels == 1]
                neg_inputs = all_inputs[all_labels == 0]
                neg_feats = all_feats[all_labels == 0]
                pos_idxs = np.random.permutation(range(len(pos_inputs)))[:self.test_batch_size // 2]
                neg_idxs = np.random.permutation(range(len(neg_inputs)))[:self.test_batch_size // 2]
                pos_idxs_feat = pos_idxs
                neg_idxs_feat = neg_idxs
                # Concatenate positives and negatives
                X = np.vstack([pos_inputs[pos_idxs], neg_inputs[neg_idxs]])
                X_feats = np.vstack([pos_feats[pos_idxs_feat], neg_feats[neg_idxs_feat]])
                y = np.hstack([np.ones(len(pos_idxs)), np.zeros(len(neg_idxs))])
            else:
                idxs = np.random.permutation(range(len(all_inputs)))[:self.test_batch_size]
                X, y, X_feats = all_inputs[idxs], all_labels[idxs], all_feats[idxs]
        elif self.test_strategy == 'cycle':
            # Cycle through the test set in batches, keeping track of where we are for each query point
            i = self._cycle_indices.get(point_index, 0)
            start, end = i * self.test_batch_size, (i + 1) * self.test_batch_size
            X, y, X_feats = self._test_inputs[start:end], self._test_labels[start:end], self._test_features[start:end]
            self._cycle_indices[point_index] = (i + 1) % self._num_batches
        else:
            raise Exception(f"Unrecognized test_strategy: '{test_strategy}'")

        ## Weight each point in the outer loss according to desired settings
        weights = np.ones(len(X))
        if self.dist_weight_thresh:
            # Weight points exponentially lower as distance from the query point increases.
            # (since the goal is just to have good finetuning accuracy on the query point)
            # weights = self.kernel.embed(query_point, X) #np.exp(-np.linalg.norm(self.weight_embedding(query_point) - self.weight_embedding(X), axis=-1) * 2.3 / self.dist_weight_thresh)
            weights = np.exp(-np.linalg.norm(query_feat - X_feats, axis=-1)*2.3/self.dist_weight_thresh)

        if self.test_strategy in ['sample', 'cycle'] and self.include_query_point:
            # Include query point in every batch, but downweight it
            X, y = np.vstack([X, query_point[None]]), np.hstack([y, proposed_label])
            weights = np.hstack((weights, self.query_point_weight * 1. / self._num_batches))

        if self.mixup_alpha:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(len(X),) + (1,) * (len(X.shape) - 1))
            idxs = np.random.permutation(range(len(X)))
            X2, y2 = X[idxs], y[idxs]
            X, y = (lam * X + (1 - lam) * X2).astype(np.float32), lam.squeeze() * y + (1 - lam.squeeze()) * y2

        y = np.hstack((y[:,None], weights[:,None]))

        return self._to_tensor_task(index, X, y)

    def _to_tensor_task(self, index, X, y):
        # Before converting to tensor: if `concat_query_point` is on,
        # concatenate the query point to each of the inputs.
        # Makes the metalearner aware of the task at hand when it's trying to adapt
        if self.concat_query_point:
            point_index = index % (self.num_tasks // self.num_classes)
            query_point = self._train_inputs[point_index]
            X = np.hstack((X, np.repeat(query_point[None], len(X), axis=0)))

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)
        task = TensorTask(X, y, index, self.num_classes)
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)
        return task

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        point_index = index % (self.num_tasks // self.num_classes)
        proposed_label = index % self.num_classes

        # For training, use only the query point; for testing, use the entire dataset
        # (with query point appended)
        return OrderedDict([
            ('train', self._get_train_task(index, point_index, proposed_label)), 
            ('test', self._get_test_task(index, point_index, proposed_label))
        ])

class Toy2D(NML):
    input_dim   = 2
    num_classes = 2
    plottable   = True

    @property
    def dataset(self):
        return self.X.astype(np.float32), self.y.astype(np.float32)
    
    def plot(self):
        raise NotImplementedError()


class NMLToy2D(Toy2D):
    """
    A toy NML meta-learning problem created from a dataset of (x, y) states and their positive/negative labels, 
    visited during a run of VICE on Maze-v0.
    """
    def __init__(
        self, 
        filename='vice-states.npy', 
        target_pos=[3., 3.], 
        replay_pool_size=100, 
        clip_length=100, 
        from_beginning=False, 
        include_positives=True,
        shuffle=True,
        test_strategy='all',
        test_batch_size=0,
    ):
        data, targets = get_dataset_from_file(filename, target_pos, replay_pool_size, clip_length, 
            from_beginning, include_positives, shuffle)
        negative_data, negative_targets = data[targets == 0], targets[targets == 0]

        super(NMLToy2D, self).__init__((negative_data, negative_targets), (data, targets), 
            test_strategy=test_strategy, test_batch_size=test_batch_size, num_classes=2, 
            transform=ToTensor1D(), target_transform=ToTensor1D(dtype='long'))

def get_dataset_from_file(
    filename='vice-states.npy', 
    target_pos=[3., 3.], 
    replay_pool_size=100, 
    clip_length=100, 
    from_beginning=False, 
    include_positives=True,
    shuffle=True
):
    print("from beginning:", from_beginning)
    replay_pool_states = np.load(filename)[-int(replay_pool_size):] if not from_beginning else np.load(filename)[:int(replay_pool_size)]
    print(f"Replay pool has {len(replay_pool_states)} states, each trajectory clipped at {clip_length}")

    if clip_length < 100:
        replay_pool_states = replay_pool_states.reshape(-1, 100, 2).transpose((0, 2, 1))[...,:clip_length].transpose((0, 2, 1)).reshape(-1, 2)
    
    negatives = replay_pool_states if not shuffle else np.random.permutation(replay_pool_states)
    positives = np.repeat(np.array(target_pos)[None], len(replay_pool_states), axis=0)

    if include_positives:
        # Interleave negative and positive states
        data = np.empty((len(negatives) + len(positives), 2), dtype=negatives.dtype)
        data[0::2] = negatives
        data[1::2] = positives
        
        # Label visited states as negatives, goals as positives
        targets = np.empty(len(negatives) + len(positives), dtype=np.long)
        targets[0::2] = 0
        targets[1::2] = 1
    else:
        data = negatives
        targets = np.zeros(len(negatives), dtype=np.long)

    return data, targets


def plot_positives(ax, positives):
    """
    Plots a batch of positive examples as stars
    """
    ax.scatter(positives[:,0], positives[:,1], label='positive', marker='*', color='g', s=100)
    
def plot_negatives(ax, negatives, alpha=1):
    """
    Plots a batch of negative examples as 'x's
    """
    ax.scatter(negatives[:,0], negatives[:,1], label='negative', marker='x', color='r', s=100, alpha=alpha)
    
def plot_examples(ax, positives, negatives, alpha=1, negative_below=True):
    if negative_below:
        plot_negatives(ax, negatives, alpha=alpha)
    plot_positives(ax, positives)
    if not negative_below:
        plot_negatives(ax, negatives, alpha=alpha)
 

class NoisyDuplicatesProblem(Toy2D):
    name = "noisy-duplicates"

    def __init__(self, locations, noise_std=0, seed=42):
        np.random.seed(seed)
        self.locations = locations
        self.noise_std = noise_std
        self.X, self.y = self.create_dataset(locations, noise_std=noise_std)
        super(NoisyDuplicatesProblem, self).__init__((self.X, self.y), (self.X, self.y), num_classes=2, 
            transform=ToTensor1D(), target_transform=ToTensor1D(dtype='long'))
        
    def get_example(self):
        return self.create_dataset(self.locations, noise_std=self.noise_std)

    def create_dataset(self, locations, noise_std=0):
        X, y = [], []
        for position, num_negatives, num_positives in locations:
            inputs = np.repeat(np.array(position, np.float32)[None], num_negatives + num_positives, axis=0)
            inputs += np.random.normal(0, noise_std, size=inputs.shape)
            labels = np.array([0] * num_negatives + [1] * num_positives)
            X += inputs.tolist()
            y += labels.tolist()
        
        return np.array(X), np.array(y)
    
    def plot(self, ax):
        px, py = self.X[self.y == 1], self.y[self.y == 1]
        nx, ny = self.X[self.y == 0], self.y[self.y == 0]
        ax.axis('square')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        # ax.invert_yaxis()
        plot_examples(ax, px, nx, negative_below=False)


def get_rings_dataset(inner_radius=0.3, inner_std=0.05, num_inner=150, 
                      outer_radius=1, outer_std=0.03, num_outer=150, shuffle=True):
    inner_ring = get_ring(inner_radius, inner_std, num_inner)
    outer_ring = get_ring(outer_radius, outer_std, num_outer)
    X = np.vstack((inner_ring, outer_ring))
    y = np.hstack((np.ones(num_inner), np.zeros(num_outer)))
    if shuffle:
        idxs = np.random.permutation(len(X))
        X, y = X[idxs], y[idxs]
    return X, y
    
def get_ring(radius, radius_std, num):
    angles = np.random.uniform(0, 2 * np.pi, num)
    radii = np.random.normal(radius, radius_std, num)
    xy = np.hstack(((radii * np.cos(angles))[:,None], (radii * np.sin(angles))[:,None]))
    return xy

class LinearDecisionBoundary(Toy2D):
    name = "linear-decision-boundary"

    def __init__(self):
        self.X = np.random.uniform(low = [-4, -4], high = [4, 4], size = (300, 2))
        self.y = (self.X[:,0] < 0).astype(int)
        super(LinearDecisionBoundary, self).__init__(train_data=(self.X, self.y))
        # y = np.logical_and(-5 < X[:,0], X[:,0] < 5)
    
    def get_example(self, num_examples = 1):
        points = np.random.uniform(low = [-4, -4], high = [4, 4], size = (num_examples, 2))

        y = (point[:,0] < 0).astype(int)
        
        return point, y
    
    def plot(self, ax):
        positives = self.X[self.y == 1]
        negatives = self.X[self.y == 0]
        ax.set_aspect('equal')
        ax.scatter(positives[:,0], positives[:,1], color='tab:orange')
        ax.scatter(negatives[:,0], negatives[:,1], color='tab:green') 


class TwoRingProblem(Toy2D):
    name = "two-ring-problem"

    def __init__(self, **kwargs):
        self.ring_params = kwargs
        self.X, self.y = get_rings_dataset(**kwargs)
        super(TwoRingProblem, self).__init__(train_data=(self.X, self.y))
    
    def get_example(self):
        return get_rings_dataset(**self.ring_params)
    
    def plot(self, ax=None):
        positives = self.X[self.y == 1]
        negatives = self.X[self.y == 0]
        ax = ax or plt.gca()
        ax.set_aspect('equal')
        ax.scatter(positives[:,0], positives[:,1], color='tab:orange')
        ax.scatter(negatives[:,0], negatives[:,1], color='tab:green')

class ThreeClassProblem(Toy2D):
    name = "three-class-problem"
    num_classes = 3

    def __init__(self):
        self.X = np.random.uniform(-4, 4, size = (300, 2))
        self.y = np.zeros(300)

        for i, point in enumerate(self.X):
            self.y[i] = self.classify(point)
        
        super(ThreeClassProblem, self).__init__(train_data=(self.X, self.y))
    
    def classify(self, point):
        if point[0] > 1.33:
            return 0
        elif point[0] < -1.33:
            return 1
        else:
            return 2
        
    def get_example(self):
       point = np.random.uniform(-4, 4, size = (2))

       return point, self.classify(point)

    
    def plot(self, ax):
       positives = self.X[self.y == 1]
       negatives = self.X[self.y == 0]
       neutrals = self.X[self.y == 2]
       ax.set_aspect('equal')
       ax.scatter(positives[:,0], positives[:,1], color='tab:orange')
       ax.scatter(negatives[:,0], negatives[:,1], color='tab:green')
       ax.scatter(neutrals[:,0], neutrals[:,1], color='tab:red')


class SupervisedDuplicates(NML):
    def __init__(self, ds, classes):
        """
        Parameters
        ----------
        ds: a torchvision dataset 
        classes : dict
            A dictionary whose keys are digits and whose entries are the desired number of that class.
        """
        # TODO: make batch size a paramater? is this even used
        self.batch_size = 64
        if not os.path.exists('./data'):
            os.mkdir('./data')
        if not os.path.exists('./data/MNIST'):
            os.mkdir('./data/MNIST')
        self.loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.batch_size, shuffle=True)
        
        self.X = []
        self.y = []
        self.classes = classes
        self.num_classes = len(self.classes)
        
        for i in range(10):
            if i not in self.classes:
                self.classes[i] = 0
        
        self.batched = {x: 0 for x in range(10) }

        self.generate_examples()


        super(SupervisedDuplicates, self).__init__((self.X, self.y), (self.X, self.y), num_classes=self.num_classes, 
            #transform=ToTensor1D(), target_transform=ToTensor1D(dtype='long')
            )
    
    def num_left(self, digit_class):
        return self.classes[digit_class] - self.batched[digit_class]
    
    def generate_examples(self):
        while self.classes != self.batched:
            self.process_batch()
        
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y).float()
    
    def process_batch(self):
        X, ys = self.get_batch()

        for i, x in enumerate(X):
            digit_class = ys[i].item()

            if self.num_left(digit_class) > 0:
                self.X.append( x[0].numpy().flatten() )
                self.y.append(digit_class)
                self.batched[digit_class] += 1

    def get_test(self):
        X, ys = self.get_batch()
        xs = []
        y = []

        for i, x in enumerate(X):
            digit_class = ys[i].item()

            if digit_class == 0 or digit_class == 1:
                xs.append( x[0].numpy().flatten() )
                y.append(digit_class)
        
        return xs, y


    def get_example(self, num_examples, digit_classes = None):
        X, ys = self.get_batch()
        examples = []

        i = 0
        while len(examples) < num_examples:
            if ys[i] in digit_classes:
                examples.append((X[i][0], ys[i]))
            i += 1
        
        return examples
    
    @property
    def keys(self):
        return [key for key in self.classes.keys() if self.classes[key] > 0]
    
    def get_batch(self):
        # does this start a new iterator each time? 
        # I think this is fine since the loader reshuffles each time, but is very sketchy
        return next(iter(self.loader))
         
class MNISTDuplicates(SupervisedDuplicates):

    def __init__(self, classes, split='train', path='datasets/'):
        """
        Parameters
        ----------

        classes : dict
            A dictionary whose keys are digits and whose entries are the desired number of that class.
        """
        self.batch_size = 64
        self.input_dim = 28 * 28
        if split == 'train':
            train = True
        elif split == 'test':
            train = False
        else:
            raise Exception("Invalid split")
        self.ds = torchvision.datasets.MNIST(path, train=train, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        super(MNISTDuplicates, self).__init__(self.ds, classes)

        """
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
                batch_size=self.batch_size, shuffle=True)
        
        self.X = []
        self.y = []
        self.classes = classes
        self.num_classes = len(self.classes)
        
        for i in range(10):
            if i not in self.classes:
                self.classes[i] = 0
        
        self.batched = {x: 0 for x in range(10) }

        self.generate_examples()


        super(MNISTDuplicates, self).__init__(self.ds, classes)
        # super(MNISTDuplicates, self).__init__((self.X, self.y), (self.X, self.y), num_classes=self.num_classes, 
            # #transform=ToTensor1D(), target_transform=ToTensor1D(dtype='long')
            # )
        """

class MNISTCDuplicates(SupervisedDuplicates):

    def __init__(self, classes, corruption, split='test', path='datasets/'):
        """
        Parameters
        ----------

        classes : dict
            A dictionary whose keys are digits and whose entries are the desired number of that class.
        """
        self.batch_size = 64
        self.input_dim = 28 * 28
        ds_name = "mnist_corrupted/{}".format(corruption)
        extra_transforms = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
        self.ds = load_tfds_as_torchvision(ds_name, split, extra_transforms, path)
        
        # torchvision.datasets.CIFAR10('datasets/', train=True, download=True,
                             # transform=torchvision.transforms.Compose([
                               # torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                                 # (0.1307,), (0.3081,))
                             # ]))
        super(MNISTCDuplicates, self).__init__(self.ds, classes)
    
class CIFAR10Duplicates(SupervisedDuplicates):
    def __init__(self, classes, split='train', path='datasets/'):
        """
        Parameters
        ----------

        classes : dict
            A dictionary whose keys are digits and whose entries are the desired number of that class.
        """
        self.batch_size = 64
        self.input_dim = 3 * 32 * 32
        is_train = split == 'train'
        self.ds = torchvision.datasets.CIFAR10(path, train=is_train, download=True,
                             transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               # torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                                 # (0.1307,), (0.3081,))
                             ]))
        super(CIFAR10Duplicates, self).__init__(self.ds, classes)

class CIFAR10CDuplicates(SupervisedDuplicates):
    def __init__(self, classes, corruption, split='train', path='datasets/'):
        """
        Parameters
        ----------

        classes : dict
            A dictionary whose keys are digits and whose entries are the desired number of that class.
        """
        self.batch_size = 64
        self.input_dim = 3 * 32 * 32
        ds_name = "cifar10_corrupted/{}".format(corruption)
        extra_transforms = torchvision.transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           ])
        self.ds = load_tfds_as_torchvision(ds_name, split, extra_transforms, path)
        super(CIFAR10CDuplicates, self).__init__(self.ds, classes)
