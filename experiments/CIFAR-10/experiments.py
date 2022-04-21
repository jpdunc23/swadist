"""Run experiments for CIFAR-10.

"""

import os
import time
import argparse
import datetime

from copy import deepcopy

import numpy as np

import torch
import torch.multiprocessing as mp

from swadist.utils import spawn_fn


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', help='Path to data directory.')
    parser.add_argument('--logdir', help='Path to directory where the Tensorboard logs are saved.')
    parser.add_argument('--savedir', help='Path to directory where state_dicts are saved.')
    args = parser.parse_args()
    datadir = args.datadir
    logdir = args.logdir
    savedir = args.savedir

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('Using cuda')
    else:
        print('Using cpu')

    # mp.spawn may throw an error without this
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # number of model replicas
    world_size = 2

    common_kwargs = {
        'dataloader_kwargs': {
            'dataset': 'cifar10',
            'root_dir': datadir,
            'num_workers': 2,
        },
        'model_kwargs': {
            'n_classes': 10,
            'in_kernel_size': 3,
            'stack_sizes': [1, 1, 1],
            'batch_norm': False,
        },
        'optimizer_kwargs': {
            'nesterov': True,
        },
        'trainer_kwargs': {
            'log': True,
            'log_dir': logdir,
            'save_dir': savedir,
            'n_print': 10,
        },
        'train_kwargs': {
            'codist_kwargs': {
                'sync_freq': 50,
                'transform': 'softmax',
            },
            'validate_per_epoch': 4,
            'save': True,
        },
        'scheduler_kwargs': {
            'alpha': 0.25,
            'decay_epochs': 15,
        },
        'swa_scheduler_kwargs': {
            'anneal_strategy': 'cos',
            'anneal_epochs': 3,
        },
    }

    # batch_size
    # batch_size = [2**i for i in range(1, 14)]
    batch_size = [2**i for i in range(6, 14)]

    # initial lr and momentum
    # lr0 = 2**np.array([-8.5, -12.5, -5., -8.5, -7., -5., -5., -5., -6., -5, -7, -4, -6])
    lr0 = 2**np.array([-5., -5., -5., -6., -5., -7., -4., -6.])

    # momentum = [.675, .98, .63, .97, .975, .95, .97, .975, .98, .975, .98, .97, .975]
    momentum = [.95, .97, .975, .98, .975, .98, .97, .975]

    assert len(lr0) == len(batch_size), 'lr0 has the wrong size'
    assert len(momentum) == len(batch_size), 'momentum has the wrong size'

    seed = int((datetime.date.today() - datetime.date(2022, 4, 11)).total_seconds())
    print(f'seed: {seed}')

    methods = ['sgd', 'swa', 'codist', 'codist-swa', 'swadist']
    method_kwargs = { method: deepcopy(common_kwargs) for method in methods }

    method_kwargs['sgd']['dataloader_kwargs'].update({ 'data_parallel': True })
    method_kwargs['sgd']['trainer_kwargs'].update({ 'name': 'sgd' })
    method_kwargs['sgd']['train_kwargs'].update({ 'epochs_sgd': 15, 'codist_kwargs': None })

    method_kwargs['swa']['dataloader_kwargs'].update({ 'data_parallel': True })
    method_kwargs['swa']['trainer_kwargs'].update({ 'name': 'swa' })
    method_kwargs['swa']['train_kwargs'].update({ 'epochs_sgd': 10, 'epochs_swa': 5, 'codist_kwargs': None })

    method_kwargs['codist']['dataloader_kwargs'].update({ 'split_training': True })
    method_kwargs['codist']['trainer_kwargs'].update({ 'name': 'codist' })
    method_kwargs['codist']['train_kwargs'].update({ 'epochs_sgd': 5, 'epochs_codist': 10 })

    method_kwargs['codist-swa']['dataloader_kwargs'].update({ 'split_training': True })
    method_kwargs['codist-swa']['trainer_kwargs'].update({ 'name': 'codist-swa' })
    method_kwargs['codist-swa']['train_kwargs'].update({ 'epochs_sgd': 5, 'epochs_codist': 5, 'epochs_swa': 5 })

    method_kwargs['swadist']['dataloader_kwargs'].update({ 'split_training': True })
    method_kwargs['swadist']['trainer_kwargs'].update({ 'name': 'swadist' })
    method_kwargs['swadist']['train_kwargs'].update({ 'epochs_sgd': 5,
                                                      'epochs_codist': 5,
                                                      'epochs_swa': 5,
                                                      'swadist': True })

    for bs, lr, mo in zip(batch_size, lr0, momentum):

        for method, kwargs in method_kwargs.items():

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            kwargs['dataloader_kwargs']['batch_size'] = bs // world_size
            kwargs['optimizer_kwargs'].update({ 'lr': lr, 'momentum': mo })
            kwargs['swa_scheduler_kwargs']['swa_lr'] = lr / 10

            if method in ['sgd', 'codist']:
                swa_scheduler_kwargs = None
            else:
                swa_scheduler_kwargs = kwargs['swa_scheduler_kwargs']

            args = (world_size,
                    kwargs['dataloader_kwargs'],
                    kwargs['model_kwargs'],
                    kwargs['optimizer_kwargs'],
                    kwargs['trainer_kwargs'],
                    kwargs['train_kwargs'],
                    kwargs['scheduler_kwargs'],
                    swa_scheduler_kwargs,
                    seed)

            tic = time.perf_counter()

            mp.spawn(spawn_fn, args=args, nprocs=world_size, join=True)

            print(f'time elapsed: {(time.perf_counter() - tic) / 60:.2f}m')

            seed += 1
