'''Functions to load datasets
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/notebooks/cifar10/cifar10.py
'''

import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


def get_dataloaders(dataset='cifar10',
                    root_dir='./data',
                    download=True,
                    validation=True,
                    test=False,
                    split_training=False,
                    world_size=1,
                    rank=0,
                    data_parallel=False,
                    getter_kwargs=None,
                    **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset: str
        Name of the dataset to load.
    root_dir: str
        Path to the dataset root.
    validation: bool
        If True, load the validation set.
    test: bool
        If True, load the test set.
    split_training: bool
        If True, use `torch.utils.data.SubsetRandomSampler`. Must be False if
        `data_parallel` is True.
    data_parallel:
        If True, use `torch.utils.data.distributed.DistributedSampler`. Must be False if
        `split_training` is True.
    getter_kwargs: dict
        Additional keyword arguments to pass to `DataLoader`. Default values are modified.
    kwargs:
        Additional keyword arguments to `DataLoader`. Should not include 'sampler'.
    """

    if data_parallel and split_training:
        raise ValueError('Only one of split_training and data_parallel can be True')

    data_getters = {
        # should return a list of datasets
        'cifar10' : get_cifar10,
    }

    if getter_kwargs is None:
        getter_kwargs = {}

    getter_kwargs.update({
        'root_dir': root_dir,
        'download': download,
        'validation': validation,
        'test': test
    })

    datasets = data_getters[dataset](**getter_kwargs)

    # create training data sampler
    train_sampler = None
    if split_training:
        indices = np.array_split(np.arange(len(datasets[0])), world_size)
        train_sampler = SubsetRandomSampler(indices[rank])
        print(f'Using SubsetRandomSampler with samples '
              f'{min(indices[rank])} to {max(indices[rank])}')
    elif data_parallel:
        train_sampler = DistributedSampler(datasets[0])
        print(f'Using DistributedSampler')
    else:
        train_sampler = RandomSampler(datasets[0])
        print(f'Using RandomSampler')

    loaders = []
    for i, dset in enumerate(datasets):
        loaders.append(
            DataLoader(dset,
                       sampler=None if i != 0 else train_sampler,
                       **kwargs)
        )

    print(f'Number of training batches: {len(loaders[0])}')

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


def get_cifar10(root_dir='./data',
                download=False,
                validation=True,
                validation_prop=0.1,
                test=True,
                **kwargs):
    datasets = []

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
        datasets.append(valid_dataset)
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

    datasets = [train_dataset] + datasets

    # download/load test data
    if test:
        test_dataset = CIFAR10(root=root_dir,
                               train=False,
                               transform=transform,
                               download=download)
        test_dataset.inv_transform = inv_transform
        datasets.append(test_dataset)

    return datasets
