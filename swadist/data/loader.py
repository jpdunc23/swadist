'''Functions to load datasets
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/notebooks/cifar10/cifar10.py
'''

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataloaders(dataset='cifar10', root_dir='./data', download=False,
                    validation_split=False, data_parallel=False, cuda=False,
                    shuffle=True, num_workers=1, pin_memory=True, batch_size=64, **kwargs):
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
    sampler = DistributedSampler(dataset[0]) if data_parallel else None
    shuffle = shuffle and not data_parallel # only shuffle if not using DistributedSampler
    loaders = [
        DataLoader(datasets[i],
                   batch_size=batch_size,
                   shuffle=shuffle and i==0,
                   sampler=sampler,
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
    dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                           train=True,
                                           download=download)

    # calculate mean / std of channels over the training data
    mean, std = per_channel_mean_and_std(dataset)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    inv_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0., 0., 0.], 1. / std),
        torchvision.transforms.Normalize(-mean, [1., 1., 1.])
    ])

    if validation_split:
        # hold out 10% for validation set
        valid_dataset = dataset
        valid_dataset.data = valid_dataset.data[45000:]
        valid_dataset.transform = transform
        valid_dataset.inv_transform = inv_transform

        # get the remaining training data
        dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                               train=True,
                                               transform=transform)
        dataset.data = dataset.data[:45000]
        dataset.inv_transform = inv_transform

    # download/load test data
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                train=False,
                                                transform=transform,
                                                download=download)
    test_dataset.inv_transform = inv_transform

    if validation_split:
        return dataset, valid_dataset, test_dataset
    return dataset, test_dataset
