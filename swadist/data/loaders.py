'''Functions to load datasets
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/notebooks/cifar10/cifar10.py
'''

import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .cifar import get_cifar10


__all__ = ['get_dataloaders']


"""Dictionary of data getters defined in `.data`. Each of the functions in this
dict should be able to accept the parameters 'root_dir', 'download',
'validation', and 'test' and should return a dict of datasets with keys among
'train' (required), 'validation', and 'test'.

"""
_data_getters = {
    'cifar10' : get_cifar10,
}


def get_dataloaders(dataset,
                    batch_size,
                    root_dir='./data',
                    download=False,
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
    batch_size: int
        Passed to `DataLoader`.
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
        Additional keyword arguments to pass to data getter function. Default values are modified.
    **kwargs:
        Additional keyword arguments to `DataLoader`. If `sampler` is included, then
        `split_training` and `data_parallel` should be False.
    """

    if data_parallel and split_training:
        raise ValueError('Only one of split_training and data_parallel can be True')

    if getter_kwargs is None:
        getter_kwargs = {}

    getter_kwargs.update({
        'root_dir': root_dir,
        'download': download,
        'validation': validation,
        'test': test
    })

    # get dict of datasets
    datasets = _data_getters[dataset](**getter_kwargs)

    # create training data sampler
    train_sampler = kwargs.get('sampler', None)

    if split_training:
        indices = np.array_split(np.arange(len(datasets['train'])), world_size)
        train_sampler = SubsetRandomSampler(indices[rank])
        print(f'Using SubsetRandomSampler with samples '
              f'{min(indices[rank])} to {max(indices[rank])}')

    elif data_parallel:
        train_sampler = DistributedSampler(datasets['train'])
        print(f'Using DistributedSampler')

    elif train_sampler is None:
        train_sampler = RandomSampler(datasets['train'])
        print(f'Using RandomSampler')

    loaders = {}

    for name, dset in datasets.items():
        loaders[name] = DataLoader(
            dset,
            batch_size=batch_size,
            sampler=None if name != 'train' else train_sampler,
            **kwargs
        )

    print(f'Number of training samples: {len(datasets["train"])}')
    print(f'Number of training batches: {len(loaders["train"])}\n')

    # convert loaders to sorted list
    loaders = [loaders[k] for k in ['train', 'validation', 'test'] if k in loaders.keys()]

    if len(loaders) == 1:
        return loaders[0]

    return loaders
