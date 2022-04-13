"""Load and pre-process CIFAR data.
"""

import numpy as np

from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from .utils import per_channel_mean_and_std


__all__ = ['get_cifar10']


def get_cifar10(root_dir='./data',
                download=False,
                validation=True,
                validation_prop=0.1,
                test=True,
                **kwargs):
    datasets = {}

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
        valid_dataset = Subset(
            dataset=CIFAR10(root=root_dir,
                            train=True,
                            transform=transform,
                            download=False),
            indices=valid_idx
        )
        valid_dataset.inv_transform = inv_transform
        datasets['validation'] = valid_dataset
    else:
        train_idx = list(range(len(dataset)))

    train_dataset = Subset(
        dataset=CIFAR10(root=root_dir,
                        train=True,
                        transform=transform,
                        download=False),
        indices=train_idx
    )
    train_dataset.inv_transform = inv_transform

    datasets['train'] = train_dataset

    # download/load test data
    if test:
        test_dataset = CIFAR10(root=root_dir,
                               train=False,
                               transform=transform,
                               download=download)
        test_dataset.inv_transform = inv_transform
        datasets['test'] = test_dataset

    return datasets
