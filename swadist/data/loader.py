'''Functions to load datasets
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/notebooks/cifar10/cifar10.py
'''

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

def get_dataloaders(dataset='cifar10', root_dir='./data', download=True,
                    validation=True, validation_prop=.1, test=True,
                    data_parallel=False, cuda=False, shuffle=True,
                    num_workers=1, pin_memory=True, batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset: str
        Name of the dataset to load.
    root_dir: str
        Path to the dataset root.
    validation: bool
        If True, load the validation set.
    validation_prop: float
        The proportion of the training set to use for validation.
    test: bool
        If True, load the test set.
    data_parallel: bool
        If True, use DistributedSampler.
    cuda: bool
        If True, using cuda.
    kwargs:
        Additional arguments to `DataLoader`. Default values are modified.
    """
    data_getters = {
        # should return tuple of (datasets, samplers)
        'cifar10' : get_cifar10,
    }
    datasets, samplers = data_getters[dataset](root_dir, download=download,
                                               validation=validation,
                                               validation_prop=validation_prop,
                                               test=test)
    pin_memory = pin_memory and cuda # only pin if not using CPU
    # samplers[0] = DistributedSampler(dataset[0]) if data_parallel else samplers[0]

    loaders = []
    for i, dset in enumerate(datasets):
        if dset is not None:
            loaders.append(DataLoader(
                dset,
                batch_size=batch_size,
                shuffle=shuffle and samplers[i] is None,
                sampler=samplers[i],
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs
            ))
    if len(loaders) == 1:
        return loaders[0]
    return loaders


def per_channel_mean_and_std(dataset, max_val=255.):
    """Returns the mean and standard deviation of each channel across images in `dataset`.
    """
    data = dataset.data / max_val
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std


def get_cifar10(root_dir='./data', download=False,
                validation=True, validation_prop=.1, test=True,
                **kwargs):

    # download the training data if needed
    dataset = CIFAR10(root=root_dir, train=True, download=download)

    # calculate mean / std of channels over the training data
    mean, std = per_channel_mean_and_std(dataset)
    transform = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])
    inv_transform = Compose([
        Normalize([0., 0., 0.], 1. / std),
        Normalize(-mean, [1., 1., 1.])
    ])
    if validation:
        split = int(np.floor(len(dataset) * (1 - validation_prop)))
        indices = list(range(len(dataset)))
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_dataset = CIFAR10(
            root=root_dir, train=True,
            download=False, transform=transform,
        )
        train_dataset.inv_transform = inv_transform

        valid_dataset = CIFAR10(
            root=root_dir, train=True,
            download=False, transform=transform,
        )
        valid_dataset.inv_transform = inv_transform
    else:
        train_dataset = CIFAR10(
            root=root_dir, train=True,
            download=False, transform=transform,
        )
        train_sampler = None
        valid_dataset = None
        valid_sampler = None

    # download/load test data
    if test:
        test_dataset = CIFAR10(root=root_dir,
                               train=False,
                               transform=transform,
                               download=download)
        test_dataset.inv_transform = inv_transform
    else:
        test_dataset = None

    datasets = (train_dataset, valid_dataset, test_dataset)
    samplers = (train_sampler, valid_sampler, None)
    return datasets, samplers
