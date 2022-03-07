'''Functions to load datasets
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/notebooks/cifar10/cifar10.py
'''

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

def get_dataloaders(dataset='cifar10', root_dir='./data', download=False,
                    validation_split=False, validation_prop=.1,
                    data_parallel=False, cuda=False, shuffle=True,
                    num_workers=1, pin_memory=True, batch_size=64, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset: str
        Name of the dataset to load.
    root_dir: str
        Path to the dataset root.
    validation_split: bool
        If True, the dataset will be loaded using a three-way split.
    data_parallel: bool
        If True, use DistributedSampler.
    cuda: bool
        If True, using cuda.
    kwargs:
        Additional arguments to `DataLoader`. Default values are modified.
    """
    dataset_getters = {
        'cifar10' : get_cifar10
    }
    datasets = dataset_getters[dataset](root_dir, download, validation_split)

    pin_memory = pin_memory and cuda # only pin if not using CPU
    # sampler = DistributedSampler(dataset[0]) if data_parallel else None
    # only shuffle if not using DistributedSampler
    shuffle = shuffle and not (data_parallel or validation_split)

    if validation_split:
        split = int(np.floor(len(datasets[0]) * (1 - validation_prop)))
        indices = list(range(len(datasets[0])))
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(datasets[0],
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  **kwargs)
        valid_loader = DataLoader(datasets[1],
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  **kwargs)
        test_loader = DataLoader(datasets[2],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 **kwargs)
        return train_loader, valid_loader, test_loader
    loaders = [
        DataLoader(datasets[i],
                   batch_size=batch_size,
                   shuffle=shuffle and i==0,
                   # sampler=sampler,
                   num_workers=num_workers,
                   pin_memory=pin_memory,
                   **kwargs)
        for i, _ in enumerate(datasets)
    ]
    return loaders


def per_channel_mean_and_std(dataset, max_val=255.):
    """Returns the mean and standard deviation of each channel across images in `dataset`.
    """
    data = dataset.data / max_val
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std


def get_cifar10(root_dir='./data', download=False, validation_split=False):

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

    if validation_split:
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

    # download/load test data
    test_dataset = CIFAR10(root=root_dir,
                           train=False,
                           transform=transform,
                           download=download)
    test_dataset.inv_transform = inv_transform

    if validation_split:
        return train_dataset, valid_dataset, test_dataset
    return train_dataset, test_dataset
