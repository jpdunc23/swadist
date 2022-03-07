import os
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
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
    model = ResNet(in_kernel_size=3, stack_sizes=[1, 1, 1], n_classes=10,
                   batch_norm=False).to(rank)

    # epochs, scaling epochs, decay factor
    epochs, T, alpha = 25, 20, 0.25
    lr_lambda = lambda epoch: 1 - (1 - alpha)*epoch/T if epoch < T else alpha
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # load training and validation data
    train_loader, valid_loader = get_dataloaders('cifar10', root_dir=datadir,
                                                 test=False, batch_size=batch_size)

    # setup optimizer / lr sched
    lr_lambda = lambda epoch: 1 - (1 - alpha)*epoch/T if epoch < T else (alpha if epoch < epochs else lr0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0,
                                momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(model, F.cross_entropy, optimizer, scheduler,
                      log=True, log_dir=rundir, name=exper,
                      device=rank, rank=rank, world_size=world_size,
                      data_parallel=True)

    # train
    if exper == 'sgd':
        trainer.train(train_loader, valid_loader, epochs=epochs,
                      stopping_acc=stopping_acc,
                      validations_per_epoch=4)
        n_epochs = trainer.total_train_epochs


    if exper == 'swa':
        trainer.train(train_loader, valid_loader, epochs=epochs-5,
                      swa_epochs=5, stopping_acc=stopping_acc,
                      validations_per_epoch=4)
        n_epochs = trainer.total_train_epochs


    if exper == 'codist':
        trainer.train(train_loader, valid_loader, epochs=5,
                      codist_epochs=epochs-5, stopping_acc=stopping_acc,
                      validations_per_epoch=4)
        n_epochs = trainer.total_train_epochs


    elif exper == 'swadist':
        trainer.train(train_loader, valid_loader, epochs=5,
                      codist_epochs=epochs-10, swa_epochs=5,
                      stopping_acc=stopping_acc, validations_per_epoch=4)

    # record metrics
    train_loss = trainer.train_losses[n_epochs]
    valid_loss = trainer.valid_losses[n_epochs]
    train_acc = trainer.train_accs[n_epochs]
    valid_loss = trainer.valid_accs[n_epochs]
    trainer.writer.add_hparams(
        { 'lr': lr0, 'batch_size': batch_size, 'momentum': momentum,
          'stopping_acc': stopping_acc, 'T': T, 'alpha': alpha },
        {'hparam/train accuracy': train_acc , 'hparam/train loss': train_loss,
         'hparam/valid accuracy': valid_acc, 'hparam/valid accuracy': valid_acc}
    )
    trainer.writer.close()


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
    # batch_size = [2**i for i in range(1, 14)]
    batch_size = [2**i for i in range(6, 14)]

    # initial lr, scaling factor, momentum
    # lr0 = 2**np.array([-8.5, -12.5, -5., -8.5, -7., -5., -5., -5., -6., -5, -7, -4, -6])
    lr0 = 2**np.array([-5., -5., -5., -6., -5, -7, -4, -6])
    # momentum = [.675, .98, .63, .97, .975, .95, .97, .975, .98, .975, .98, .97, .975]
    momentum = [.95, .97, .975, .98, .975, .98, .97, .975]

    assert len(lr0) == len(batch_size), 'lr0 has the wrong size'
    assert len(momentum) == len(momentum), 'momentum has the wrong size'

    if world_size > 1:

        for b, l, m in zip(batch_size, lr0, momentum):
            args = (world_size, b, l, m, exper, datadir, rundir)
            mp.spawn(main, args=args, nprocs=world_size, join=True)
