import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from swadist.data.loader import get_dataloaders
from swadist.utils import Trainer, show_imgs
from swadist.models.resnet import ResNet

def main(rank, world_size, batch_size, lr0, momentum, exper, datadir, rundir, stopping_acc=0.9):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6006'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # pin process to a single device
    torch.cuda.set_device(rank)

    model = DistributedDataParallel(
        ResNet(in_kernel_size=3, stack_sizes=[1, 1, 1], n_classes=10, batch_norm=False),
        device_ids=[rank], output_device=rank
    ).cuda(rank)

    # epochs, scaling epochs, decay factor
    epochs, T, alpha = 25, 20, 0.25
    lr_lambda = lambda epoch: 1 - (1 - alpha)*epoch/T if epoch < T else alpha
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # load training and validation data
    train_loader, valid_loader = get_dataloader('cifar10', test=False, batch_size=batch_size)

    # setup optimizer / lr sched
    lr_lambda = lambda epoch: 1 - (1 - alpha)*epoch/T if epoch < T else (alpha if epoch < epochs else lr0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0,
                                momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # train SGD
    if exper == 'swadist':
        trainer = Trainer(model, F.cross_entropy, optimizer, scheduler, log=True, name='sgd-train')
        trainer.train(train_loader, valid_loader, epochs=epochs,
                      stopping_acc=stopping_acc, validations_per_epoch=4, log=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exper', help='Which experiment to run.')
    parser.add_argument('--datadir', help='Path to data directory.')
    parser.add_argument('--rundir', help='Path to directory where the Tensorboard log is saved')
    args = parser.parse_args()
    exper = args.exper
    datadir = args.datadir
    rundir = args.rundir

    world_size = torch.cuda.device_count()
    print(f'n GPUs: {world_size}')

    # batch_size
    batch_size = [2**i for i in range(1, 14)]

    # initial lr, scaling factor, momentum
    lr0 = 2**np.array([-8.5, -12.5, -5., -8.5, -7., -5., -5., -5., -6., -5, -7, -4, -6])
    momentum = [.675, .98, .63, .97, .975, .95, .97, .975, .98, .975, .98, .97, .975]

    assert len(lr0) == len(batch_size), 'lr0 has the wrong size'
    assert len(momentum) == len(momentum), 'momentum has the wrong size'

    if world_size > 1:

        for b, l, m in zip(batch_size, lr0, momentum):
            args = (world_size, b, l, m, exper, datadir, rundir)
            mp.spawn(main, args=args, nprocs=world_size, join=True)
