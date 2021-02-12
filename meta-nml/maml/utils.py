import torch
import torchvision.transforms as transforms

from collections import OrderedDict
from torchmeta.modules import MetaModule
from .custom_tensor_dataset import CustomTensorDataset


# import tensorflow_datasets as tfds
import numpy as np

def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               num_layers=None,
                               first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    num_layers = len(params.values()) if not num_layers else num_layers * 2  # (2 * num_layers) since every layer has weight & bias params
    grads = torch.autograd.grad(loss,
                                list(params.values())[-num_layers:],
                                create_graph=not first_order)
    updated_params = OrderedDict()
    # "Freeze" the earlier layers
    for name, param in list(params.items())[:-num_layers]:
        updated_params[name] = param
    # Finetune the last `num_layers` layers
    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(list(params.items())[-num_layers:], grads):
            updated_params[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(list(params.items())[-num_layers:], grads):
            updated_params[name] = param - step_size * grad
    return updated_params

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to float32 or the type specified by `dtype`.
    """
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def __call__(self, array):
        return torch.from_numpy(array.astype(self.dtype) if self.dtype else array)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def load_tfds_as_torchvision(name, split, extra_transforms, path):
    # cifar normalizations
    # extra_transforms = transforms.Compose(
        # [
            # transforms.Resize(32),
            # transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ]
    # )
    ds = tfds.load(name, data_dir=path)[split]
    # ds = tfds.load("cifar10")['test']
    np_ds = ds.as_numpy_iterator()
    images, labels = [], []
    transformed_images = []
    for i in np_ds:
        images.append(np.moveaxis(i['image'], 2, 0))
        labels.append(i['label'])
    images, labels = torch.Tensor(images) / 255, torch.LongTensor(labels)
    # images = images / 255
    all_transforms = transforms.Compose([transforms.ToPILImage(), extra_transforms])
    # all_transforms = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # for i in images:
        # transformed_images.append(all_transforms(i))
    # transformed_images = torch.stack(transformed_images)
    
    # torch_ds = CustomTensorDataset((transformed_images, labels)) # , all_transforms)
    torch_ds = CustomTensorDataset((images, labels), all_transforms)
    return torch_ds
    return {
            split: torch.utils.data.DataLoader(
                torch_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                )}
