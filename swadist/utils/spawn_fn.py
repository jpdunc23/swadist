import os

import numpy as np

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim.swa_utils import SWALR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from ..data import get_dataloaders
from ..train import Trainer
from ..optim import LinearPolyLR
from ..models import ResNet


__all__ = ['spawn_fn']


def spawn_fn(rank: int,
             world_size: int,
             dataloader_kwargs: dict,
             model_kwargs: dict,
             optimizer_kwargs: dict,
             trainer_kwargs: dict,
             train_kwargs: dict,
             scheduler_kwargs: dict=None,
             swa_scheduler_kwargs: dict=None,
             addr: str='127.0.0.1',
             port: str='6016',
             backend: str=None):
    """A function that can be passed to `torch.multiprocessing.spawn` for distributed
    training of SWADist.

    Parameters
    ----------
    rank: int
        Automatically passed to this function by `mp.spawn`
    world_size: int
        Number of distributed workers.
    dataloader_kwargs: dict
        Keyword arguments to pass to `get_dataloaders`.
    model_kwargs: dict
        Keyword arguments to pass to `ResNet`.
    optimizer_kwargs: dict
        Keyword arguments to pass to `LinearPolyLR`.
    trainer_kwargs: dict
        Keyword arguments to pass to `Trainer`.
    train_kwargs: dict
        Keyword arguments to pass to `Trainer.train`.
    scheduler_kwargs: dict, optional
        Keyword arguments to pass to `LinearPolyLR`.
    swa_scheduler_kwargs: dict, optional
        Keyword arguments to pass to `torch.optim.swa_utils.SWALR`.
    addr, port, backend: str, optional
        Used to setup the process group.

    """

    cuda = torch.cuda.is_available()

    # initialize the process group
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    if backend is None:
        backend = 'nccl' if cuda else 'gloo'

    dist.init_process_group(backend, world_size=world_size, rank=rank)

    # pin process to a single cuda device
    if cuda:
        torch.cuda.set_device(rank)
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Rank {rank} using {device} with random seed {torch.seed()}')

    # get_dataloaders
    dataloader_kwargs.setdefault('world_size', world_size)
    dataloader_kwargs.setdefault('rank', rank)
    dataloader_kwargs.setdefault('pin_memory', cuda)

    train_loader, valid_loader = get_dataloaders(**dataloader_kwargs)

    data_parallel = dataloader_kwargs.get('data_parallel', False)

    # model
    model = ResNet(**model_kwargs).to(device)
    if cuda and data_parallel and not train_kwargs.get('epochs_codist', 0) > 0:
        # codistillation is incompatible with DDP
        print('Rank {rank} using DistributedDataParallel')
        model = DistributedDataParallel(model)

    # optimizer
    optimizer_kwargs['params'] = model.parameters()
    optimizer = torch.optim.SGD(**optimizer_kwargs)

    # SGD scheduler
    if scheduler_kwargs is not None:
        scheduler_kwargs['optimizer'] = optimizer
        scheduler = LinearPolyLR(**scheduler_kwargs)
    else:
        scheduler = None

    # SWA scheduler
    if swa_scheduler_kwargs is not None:
        swa_scheduler_kwargs['optimizer'] = optimizer
        swa_scheduler_kwargs.setdefault('anneal_strategy', 'linear')
        swa_scheduler_kwargs.setdefault('anneal_epochs', 0)
        swa_scheduler = SWALR(**swa_scheduler_kwargs)
    else:
        swa_scheduler = None

    # Trainer
    trainer_kwargs['model'] = model
    trainer_kwargs['train_loader'] = train_loader
    trainer_kwargs['valid_loader'] = valid_loader
    trainer_kwargs['loss_fn'] = F.cross_entropy
    trainer_kwargs['optimizer'] = optimizer
    trainer_kwargs['scheduler'] = scheduler
    trainer_kwargs['swa_scheduler'] = swa_scheduler
    trainer_kwargs['rank'] = rank
    trainer_kwargs['device'] = device
    trainer_kwargs['world_size'] = world_size

    trainer = Trainer(**trainer_kwargs)

    # start training
    train_kwargs.setdefault('save', True)
    trainer.train(**train_kwargs)
