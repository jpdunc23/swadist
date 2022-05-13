"""Run experiments for CIFAR-10.

"""

import os
import argparse
import datetime

from copy import deepcopy

import numpy as np

import torch

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
            'num_workers': 1,
            'data_parallel': True,
            # 'pin_memory': True, # set in spawn_fn
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
            'n_print': 0,
        },
        'train_kwargs': {
            'swadist_kwargs': {
                'transform': 'softmax',
                'max_averaged': 3,
            },
            'stopping_acc': 0.7,
            'save': True,
        },
        'scheduler_kwargs': {
            'alpha': 0.25,
            'decay_epochs': 50,
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

    methods = ['swadist', 'swadist-replicas', 'codist', 'sgd']
    method_kwargs = { method: deepcopy(common_kwargs) for method in methods }

    # swadist w/o SWA replicas
    method_kwargs['swadist']['trainer_kwargs'].update({ 'name': 'swadist' })
    method_kwargs['swadist']['train_kwargs'].update({ 'epochs_sgd': 0,
                                                      'epochs_codist': 0,
                                                      'epochs_swa': 200,
                                                      'swadist': True })

    # swadist with SWA replicas
    method_kwargs['swadist-replicas']['trainer_kwargs'].update({ 'name': 'swadist-replicas' })
    method_kwargs['swadist-replicas']['train_kwargs'].update({ 'epochs_sgd': 0,
                                                               'epochs_codist': 0,
                                                               'epochs_swa': 200,
                                                               'swadist': True })
    method_kwargs['swadist-replicas']['train_kwargs']['swadist_kwargs'].update(
        { 'swa_replicas': True }
    )

    # codist
    method_kwargs['codist']['trainer_kwargs'].update({ 'name': 'codist' })
    method_kwargs['codist']['train_kwargs'].update({ 'epochs_sgd': 0,
                                                     'epochs_codist': 200,
                                                     'epochs_swa': 0 })
    method_kwargs['codist']['train_kwargs']['swadist_kwargs'] = None
    method_kwargs['codist']['train_kwargs']['codist_kwargs'] = {
        'sync_freq': 50,
        'transform': 'softmax',
    }

    # SGD
    method_kwargs['sgd']['trainer_kwargs'].update({ 'name': 'sgd' })
    method_kwargs['sgd']['train_kwargs'].update({ 'epochs_sgd': 200, 'swadist_kwargs': None })

    for bs, lr, mo in zip(batch_size, lr0, momentum):

        for method, kwargs in method_kwargs.items():

            if bs < 4096:
                continue
            else:
                kwargs['train_kwargs'][f'epochs_{what}'] = 400

            if method == 'swadist-replicas':
                what = 'swadist'
            else:
                what = method

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if what in ['codist', 'swadist']:
                # get ~3 syncs per epoch
                sync_freq = int(np.ceil(45000 / (3*bs)))
                kwargs['train_kwargs'][f'{what}_kwargs']['sync_freq'] = sync_freq

            trainer_kwargs = deepcopy(kwargs['trainer_kwargs'])
            trainer_kwargs['name'] = f'bs{bs}-' + trainer_kwargs['name']

            kwargs['dataloader_kwargs']['batch_size'] = bs // world_size
            kwargs['optimizer_kwargs'].update({ 'lr': lr, 'momentum': mo })

            args = (world_size,
                    kwargs['dataloader_kwargs'],
                    kwargs['model_kwargs'],
                    kwargs['optimizer_kwargs'],
                    trainer_kwargs,
                    kwargs['train_kwargs'],
                    kwargs['scheduler_kwargs'],
                    None, # swa_scheduler_kwargs,
                    seed)

            torch.multiprocessing.spawn(spawn_fn, args=args, nprocs=world_size, join=True)
