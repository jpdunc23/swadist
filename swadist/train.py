"""Trainer
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

import os
import pathlib
import warnings

from copy import deepcopy
from datetime import datetime

import torch
import numpy as np

from torch.nn.utils import parameters_to_vector
from torchvision.utils import make_grid
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.tensorboard import SummaryWriter

from .optim import LinearPolyLR
from .metrics import accuracy, CodistillationLoss, SWADistLoss
from .distributed import all_reduce, is_multiproc


__all__ = ['Trainer']


class Trainer():
    """
    Class to handle training of SWADist.

    Parameters
    ----------
    model: torch.nn.Module
        Model to train.
    train_loader: torch.utils.data.DataLoader
        Training set `DataLoader`.
    valid_loader: torch.utils.data.DataLoader
        Validation set `DataLoader`.
    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]
        Differentiable loss function. Should use reduction='mean'.
    optimizer: torch.optim.Optimizer
        Optimizer.
    scheduler: torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler for SGD phase.
    swa_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, dict], optional
        Either a scheduler or a dictionary of keyword arguments to pass to `SWALR`.
    device: torch.device, optional
        Device on which to run the code.
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    world_size: int
        The number of distributed workers. If greater than 1, then Trainer should be part of a
        distributed process group.
    name: str
        Name for this Trainer.
    log: bool
        If True, write metrics and plots to a `torch.utils.tensorboard.SummaryWriter`.
    log_dir: str
        Directory to use for SummaryWriter output and saving the model.
    save_dir: str
        Directory to use for saving the model after each training run.
    n_print: int
        How often to print training / validation metrics, in number of steps.

    """
    def __init__(self,
                 # training params
                 model,
                 train_loader,
                 valid_loader,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 swa_scheduler=None,
                 # compute params
                 device='cpu',
                 rank=0,
                 world_size=1,
                 # logging / saving params
                 name='trainer',
                 log=False,
                 log_dir='./runs',
                 save_dir='./state_dicts',
                 prints=True,
                 n_print=0):

        self.device = device
        self.model = model.to(self.device)
        self.swa_model = None
        self.final_model = None
        self.ddp = model.__class__.__name__ == 'DistributedDataParallel'

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_batches = len(train_loader)

        self.loss_fn = loss_fn
        self.codist_loss_fn = None
        self.swadist_loss_fn = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_scheduler = swa_scheduler

        self.rank = rank
        self.world_size = world_size

        self.n_print = n_print
        self.prints = prints

        self.name = name
        self.logs = log
        self.log_dir = log_dir

        self.save_dir = save_dir
        self.train_id = None

        # the below are set during training

        self.hparams = None

        self.epochs_sgd = 0
        self.epochs_codist = 0
        self.epochs_swa = 0

        self.in_sgd = None
        self.in_codist = None
        self.in_swadist = None
        self.in_swa = None

        self.stop_stall_n_epochs = None
        self.stopping_acc = None
        self.stop_early = None

        # dict with list entries of epoch metrics
        self.epoch_train_metrics = None
        self.epoch_valid_metrics = None

        # dict with list entries of step metrics
        self.step_train_metrics = None
        self.step_valid_metrics = None

        # current step
        self.step = None

        # current epoch
        self.epoch = None

        self.writer = None


    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)


    def train(self,
              epochs_sgd: int=10,
              epochs_codist: int=0,
              epochs_swa: int=0,
              swadist: bool=False,
              codist_kwargs: dict=None,
              swadist_kwargs: dict=None,
              stopping_acc: float=None,
              stop_stall_n_epochs: int=None,
              save: bool=False,
              save_dir: str=None):
        """Trains the model.

        Parameters
        ----------
        epochs_sgd: int, optional
            Number of epochs to train the model using vanilla SGD.
        epochs_codist: bool, optional
            Number of epochs using codistillation.
        epochs_swa: int, optional
            Number of epochs using stochastic weight averaging.
        swadist: bool, optional
            If True, use codistillation during the SWA phase.
        codist_kwargs: dict, optional
            Additional keyword arguments to pass to `CodistillationLoss` initializer.
        stopping_acc: float, optional
            Validation accuracy at which to stop training.
        stop_stall_n_epochs: int, optional
            If the mean over the last `stop_stall_n_epochs` is less than the mean
            over the previous `stop_stall_n_epochs`, then training is stopped.
        save: bool
            If True, save the final model, optimizer, and scheduler states and hyperparameters.
        save_dir: str
            Directory to use for saving the model.

        """
        self.swadist = swadist

        # save hyperparameters
        hparams = {
            'name': self.name,
            'world_size': self.world_size,
            'global_batch_size': self.train_loader.batch_size * self.world_size,
            'optimizer': type(self.optimizer).__name__,
            'loss_fn': self.loss_fn.__name__,
        }

        if self.swadist or epochs_codist > 0:
            assert not self.ddp, \
                'Codistillation is incompatible with DistributedDataParallel models.'
            assert is_multiproc(), \
                ('To run codistillation, use multiprocessing with world_size greater than 1 ',
                 '(see swadist.utils.spawn_fn).')

        self.in_codist = False
        self.in_swa = False

        self.epochs_codist = epochs_codist
        self.epochs_swa = epochs_swa

        self.stop_stall_n_epochs = stop_stall_n_epochs
        self.stopping_acc = stopping_acc
        self.stop_early = False

        self.step = 0
        self.epoch = 0

        # add the rest of the hyperparams
        hparams.update(self.optimizer.defaults)

        if self.scheduler is not None:
            hparams['scheduler'] = type(self.scheduler).__name__

            if isinstance(self.scheduler, LinearPolyLR):
                hparams['scheduler_alpha'] = self.scheduler.alpha
                hparams['scheduler_decay_epochs'] = self.scheduler.alpha

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.train_id = f'{self.name}_{timestamp}'

        # setup logging
        if self.logs:
            # create the writer
            self.writer = SummaryWriter(f'{self.log_dir}/{self.train_id}')

        total_epochs = epochs_sgd + epochs_codist + epochs_swa
        if not is_multiproc():
            print(f'Starting {total_epochs}-epoch training loop...')
            print(f'Random seed: {torch.seed()}\n')
        else:
            print(f'Worker {self.rank+1}/{self.world_size} starting '
                  f'{total_epochs}-epoch training loop...')

        if self.prints:
            swa_str = 'SWADist' if self.swadist else 'SWA'
            print(f'SGD epochs: {epochs_sgd} | '
                  f'Codistillation epochs: {epochs_codist} | '
                  f'{swa_str} epochs: {epochs_swa}')
            print(f'DistributedDataParallel: {self.ddp}')
            print(f'Stopping accuracy: {self.stopping_acc}')
            print(f'Global batch size: {hparams["global_batch_size"]}\n')

        # begin training

        self.epoch_train_metrics = {}
        self.epoch_valid_metrics = {}
        self.step_train_metrics = {}
        self.step_valid_metrics = {}

        # vanilla SGD / burn-in

        if epochs_sgd > 0:

            self._sgd(epochs_sgd)

        # codistillation

        if epochs_codist > 0:

            if codist_kwargs is None:
                codist_kwargs = {}

            self._codist(epochs_codist, codist_kwargs)

        # stochastic weight averaging

        if epochs_swa > 0:

            hparams['swadist'] = self.swadist

            if self.swa_scheduler is not None:

                if isinstance(self.swa_scheduler, dict):
                    hparams['swa_scheduler'] = 'SWALR'
                    for key, val in self.swa_scheduler.items():
                        hparams[f'swa_scheduler_{key}'] = val

                else:
                    hparams['swa_scheduler'] = type(self.swa_scheduler).__name__
                    if isinstance(self.swa_scheduler, torch.optim.swa_utils.SWALR):
                        hparams['swa_scheduler_swa_lr'] = \
                            self.swa_scheduler.optimizer.param_groups[0]['swa_lr']
                        hparams['swa_scheduler_anneal_epochs'] = self.swa_scheduler.anneal_epochs
                        hparams['swa_scheduler_anneal_strategy'] = \
                            self.swa_scheduler.anneal_func.__name__

            if swadist_kwargs is None:
                swadist_kwargs = {}

            self._swa(epochs_swa, swadist_kwargs)

        else:
            self.swa_scheduler = None

        if not self.stop_early:
            # cache the trained network
            self.final_model = self.model

        self.model = None

        if self.epochs_codist > 0:
            hparams['codist_sync_freq'] = codist_kwargs.get('sync_freq', 50)
            hparams['codist_loss_fn'] = codist_kwargs.get('loss_fn', self.loss_fn).__name__
            hparams['codist_transform'] = codist_kwargs.get('transform', None)

        if self.epochs_swa > 0 and self.swadist:
            hparams['swadist_sync_freq'] = swadist_kwargs.get('sync_freq', 50)
            hparams['swadist_loss_fn'] = swadist_kwargs.get('loss_fn', self.loss_fn).__name__
            hparams['swadist_transform'] = swadist_kwargs.get('transform', None)
            hparams['swadist_max_averaged'] = swadist_kwargs.get('max_averaged', None)

        self.hparams = hparams

        torch.cuda.synchronize()

        if self.logs:
            self._log_metrics()

        # all ranks because we all_reduce the metrics
        self._log_hparam_metrics()

        if self.logs:
            self.writer.close()

        if save:
            self.save(save_dir=save_dir)


    def _check_for_stall(self):

        if self.stop_early:
            return

        # stop current phase early if training has stalled out (avg acc of most
        # recent `stop_stall_n_epochs` epochs is less than that of n_avg+1
        # epochs ago)
        check_for_stall = (
            self.stop_stall_n_epochs is not None and
            self.stop_stall_n_epochs > 0 and
            ((self.in_sgd and
              self.epochs_sgd >= 2*self.stop_stall_n_epochs) or
             (self.in_codist and
              self.epochs_codist >= 2*self.stop_stall_n_epochs) or
             ((self.in_swa or self.in_swadist) and
              self.epochs_swa >= 2*self.stop_stall_n_epochs))
        )

        if check_for_stall:

            print(f'Rank {self.rank} checking for stall')

            n_avg = self.stop_stall_n_epochs
            acc_mean = torch.asarray(self.epoch_valid_metrics['acc'][-n_avg:]).mean()
            old_mean = torch.asarray(self.epoch_valid_metrics['acc'][-(2*n_avg):-n_avg]).mean()
            acc = acc_mean - old_mean

            print(f'Rank {self.rank} acc difference (last {n_avg} epochs vs the {n_avg} before that): {acc:.6f}')

            if not self.ddp and is_multiproc():

                # get avg mean acc difference across ranks
                _, acc = all_reduce(acc,
                                    op=torch.distributed.ReduceOp.AVG,
                                    async_op=False)

            if acc < 0:

                self.stop_early = True

                if self.prints:

                    print(f'\n\nThis phase of training stalled (mean accuracy of last {n_avg} '
                          f'epochs was {acc:.6f} less then mean of the {n_avg} before, '
                          'on average across all ranks). Moving onto the next phase.')


    def _stop_early(self):

        if self.stopping_acc is None or self.stop_early:
            # already stopped early
            return

        acc = self.epoch_valid_metrics['acc'][-1]

        if not self.ddp and is_multiproc():

            # get average validation acc across ranks
            _, acc = all_reduce(acc,
                                op=torch.distributed.ReduceOp.AVG,
                                async_op=False)

        if acc >= stopping_acc:

            self.stop_early = True

            if self.prints:
                print('\n\nValidation accuracy target reached '
                      f'(mean accuracy across ranks at epoch {self.epoch}: {acc:.6f}). '
                      'Saving current model and stopping.')

            self.model.cpu()
            self.final_model = deepcopy(self.model)
            self.model.to(self.device)


    def _sgd(self, epochs):

        self.in_sgd = True

        # epoch loop
        for _ in range(epochs):

            if self.stop_early:
                break

            self.epochs_sgd += 1

            self._train_epoch()

        self.in_sgd = False

        # in case sgd stalled out
        if self.epochs_codist + self.epochs_swa > 0:
            self.stop_early = False


    def _codist(self, epochs, codist_kwargs):
        self.in_codist = True
        self.epochs_codist = 0

        if epochs > 0 and self.prints:
            print(f'Starting codistillation phase...\n')

        loss_fn = codist_kwargs.get('loss_fn', self.loss_fn)

        # init codistillation loss
        self.codist_loss_fn = CodistillationLoss(loss_fn,
                                                 self.model,
                                                 self.rank,
                                                 self.world_size,
                                                 **codist_kwargs)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):

            if self.stop_early:
                break

            self.epochs_codist += 1

            self._train_epoch()

        self.codist_loss_fn = None
        self.in_codist = False

        # in case codist stalled out
        if self.epochs_swa > 0:
            self.stop_early = False


    def _swa(self, epochs, swadist_kwargs):

        if self.swadist:
            self.in_swadist = True
            phase_name = 'SWADist'
        else:
            self.in_swa = True
            phase_name = 'SWA'

        self.epochs_swa = 0

        if self.prints:
            print(f'Starting {phase_name} phase...\n')

        # self.model.to('cpu')

        # create the model for SWA
        if self.in_swadist:

            loss_fn = swadist_kwargs.get('loss_fn', self.loss_fn)

            # initialize self.codist_loss_fn
            self.swadist_loss_fn = SWADistLoss(loss_fn,
                                               self.model,
                                               self.rank,
                                               self.world_size,
                                               **swadist_kwargs)

            self.swa_model = self.swadist_loss_fn.swa_model

        else:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_model.update_parameters(self.model)

        # self.model.to(self.rank)

        if isinstance(self.swa_scheduler, dict):
            self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, **self.swa_scheduler)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):

            if self.stop_early:
                break

            self.epochs_swa += 1

            self._train_epoch()

        self.swadist_loss_fn = None
        self.in_swadist = False
        self.in_swa = False


    def _cache_metrics(self, metrics, what='step', stage='train'):
        metric_dict = getattr(self, f'{what}_{stage}_metrics')

        # save the step metrics
        for name, metric in metrics.items():
            metric_dict.setdefault(name, [])
            metric_dict[name].append(metric)

        metric_dict.setdefault('step', [])
        metric_dict['step'].append(self.step)

        if what != 'step':
            metric_dict.setdefault('epoch', [])
            metric_dict['epoch'].append(self.step)


    def _train_epoch(self):
        """Trains the model for one epoch.

        """
        self.model.train()
        self.epoch += 1
        loss = 0.
        acc = 0.
        metrics = {}

        if self.train_loader.sampler.__class__.__name__ == 'DistributedSampler':
            # shuffles data across epochs when using DistributedSampler
            self.train_loader.sampler.set_epoch(self.epoch)

        # training loop
        for batch_idx, (x, y) in enumerate(self.train_loader):
            self.step += 1

            x, y = x.to(self.device), y.to(self.device)

            # mean over the batch size
            step_metrics = self._train_step(x, y)

            # save metrics
            self._cache_metrics(step_metrics)

            for name, metric in step_metrics.items():
                # running sum of batch mean metrics
                metrics[name] = metrics.get(name, 0.) + metric

            train_end = batch_idx == self.n_batches - 1

            if train_end:
                for name, metric in step_metrics.items():
                    # mean of batch means
                    metrics[name] = metrics[name] / self.n_batches

            if self.n_print > 0:
                print_ = (batch_idx % self.n_print) == 0
            else:
                print_ = False

            if self.prints and (print_ or train_end):

                metrics_ = metrics if train_end else step_metrics
                metrics_str = 'Metrics (epoch mean): ' if train_end else 'Metrics (batch mean): '
                metrics_str = metrics_str + \
                    f'{self.loss_fn.__name__}={metrics_[self.loss_fn.__name__]:.6f} <> '

                metric_strs = []
                for name, metric in metrics_.items():
                    if name != self.loss_fn.__name__:
                        metric_strs.append(f'{name}={metric:.6f}')

                metrics_str += ' <> '.join(metric_strs)

                end = '\n' if train_end else ''
                print(
                    f'\rTrain epoch: {self.epoch} | {metrics_str} | '
                    f'Batch: {batch_idx + 1}/{self.n_batches} '
                    f'({100. * (batch_idx + 1) / self.n_batches:.0f}%) | '
                    f'Total steps: {self.step}',
                    end=end
                )

        # calculate epoch training metrics

        # save epoch metrics
        self._cache_metrics(metrics, what='epoch')

        # validate the epoch
        self._validate()

        if (self.in_swa or self.in_swadist)  and self.swa_scheduler is not None:
            # update running average of parameters
            self.swa_scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()


    def _train_step(self, x, y):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        x: torch.Tensor
            A batch of input observations.
        y: torch.Tensor
            A batch of corresponding outputs.
        """
        # clear gradients
        self.optimizer.zero_grad()

        # calculate the loss & gradients

        output = self.model(x)

        metrics = {
            self.loss_fn.__name__: self.loss_fn(output, y),
            'acc': accuracy(output, y),
        }

        if self.in_swadist:

            metrics['swadist_loss'] = self.swadist_loss_fn(x, output)

            # backprop on mean of losses
            (0.5*(metrics[self.loss_fn.__name__] + metrics['swadist_loss'])).backward()

        elif self.in_codist:

            metrics['codist_loss'] = self.codist_loss_fn(x, output)

            # backprop on mean of losses
            (0.5*(metrics[self.loss_fn.__name__] + metrics['codist_loss'])).backward()

        else:
            metrics[self.loss_fn.__name__].backward()

        self.optimizer.step()

        # for name, metric in metrics.items():
        #     metrics[name] = metric.item()

        return metrics


    def _validate(self, what='epoch'):
        """
        Validates the current model.

        Parameters
        ----------
        what: str
            Either 'epoch' or 'step'.

        """

        metrics = {}
        epoch_val = what == 'epoch'
        step = self.epoch if epoch_val else self.step
        prints = epoch_val and self.prints
        n_batches = len(self.valid_loader)

        if self.in_swadist and epoch_val:
            # updates self.swa_model
            self.swadist_loss_fn.update_swa_parameters()

        if self.in_swa and epoch_val:
            assert not self.in_swadist

            # update the AveragedModel and batchnorm stats before every eval
            self.swa_model.update_parameters(self.model)

            # only update the bn statistics if not already averaging buffers
            if not self.swa_model.use_buffers:
                # self.swa_model.to(self.device)
                torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, self.device)

        if self.in_swa or self.in_swadist:
            # move model to the cpu
            # self.model.to('cpu')

            # move swa_model to the gpu and eval
            # self.swa_model.to(self.device)
            self.swa_model.eval()

        else:
            self.model.eval()

        # validation loop
        with torch.inference_mode():
            for batch_idx, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                if self.in_swa:
                    output = self.swa_model(x)
                else:
                    output = self.model(x)

                # add batch step loss to the validation metrics
                metrics.setdefault(self.loss_fn.__name__, 0.)
                metrics[self.loss_fn.__name__] += self.loss_fn(output, y)#.item()
                metrics.setdefault('acc', 0.)
                metrics['acc'] += accuracy(output, y)#.item()

                # final validation batch?
                val_end = batch_idx == (n_batches - 1)

                if val_end:
                    metrics[self.loss_fn.__name__] = metrics[self.loss_fn.__name__] / n_batches
                    metrics['acc'] = metrics['acc'] / n_batches

                if self.n_print > 0:
                    print_ = (batch_idx % self.n_print) == 0
                else:
                    print_ = False

                if prints and (val_end or print_):

                    metric_strs = [('\rValidation (batch mean) | ',
                                    metrics[self.loss_fn.__name__],
                                    metrics['acc'])]

                    end = '\n' if val_end else ''

                    # only print on min rank
                    for (begin, batch_loss, batch_acc) in metric_strs:
                        print(begin,
                              f'{self.loss_fn.__name__}={batch_loss:.6f} <> '
                              f'accuracy={batch_acc:.6f} | '
                              f'Batch: {batch_idx + 1}/{n_batches} '
                              f'({100.*(batch_idx + 1) / n_batches:.0f}%)',
                              end=end)

        if prints:
            print()

        if not self.stop_early:
            # save step validation loss and acc
            self._cache_metrics(metrics, what="step", stage="valid")

        if epoch_val:
            # save and log epoch loss and acc
            self._cache_metrics(metrics, what="epoch", stage="valid")

            # check if we should stop early
            self._stop_early()
            self._check_for_stall()


    def _log_separate(self, what='epoch'):
        """Separately log training and validation metrics.
        """
        train_metrics = getattr(self, f'{what}_train_metrics')
        valid_metrics = getattr(self, f'{what}_valid_metrics')
        train_iters = train_metrics[what]
        valid_iters = valid_metrics[what]

        for name, metrics in valid_metrics.items():

            for i, _ in enumerate(metrics):

                if name != 'acc':
                    name = f'loss/{name}'

                self.writer.add_scalar(f'{what} {name}/valid', metrics[i], valid_iters[i])

        for name, metrics in train_metrics.items():

            for i, _ in enumerate(metrics):

                if name != 'acc':
                    name = f'loss/{name}'

                self.writer.add_scalar(f'{what} {name}/train', metrics[i], train_iters[i])


    def _log_together(self, what='epoch'):
        """Log training and validation metrics together.
        """
        train_metrics_dict = getattr(self, f'{what}_train_metrics')
        valid_metrics_dict = getattr(self, f'{what}_valid_metrics')
        valid_iters = valid_metrics_dict[what]

        # codist_loss and swadist_loss don't have validation metrics
        for name in valid_metrics_dict.keys():

            if not name in train_metrics_dict.keys():
                continue

            if what == 'step':
                train_metrics = [metric for step, metric in zip(train_metrics_dict['step'],
                                                                train_metrics_dict[name])
                                 if step in valid_iters]
            else:
                train_metrics = train_metrics_dict[name]

            for i, valid_metric in enumerate(valid_metrics_dict[name]):

                scalars = {
                    'valid': valid_metric,
                    'train': train_metrics[i],
                }

                if name != 'acc':
                    name = f'loss/{name}'

                self.writer.add_scalars(f'{what} {name}', scalars, valid_iters[i])


    def _log_hparam_metrics(self):
        log_dict = {}

        for what in ['step', 'epoch']:
            metrics_dict = deepcopy(getattr(self, f'{what}_valid_metrics'))

            if what == 'step':
                whats = metrics_dict.pop('step')
            else:
                whats = metrics_dict.pop('epoch')

            for name, metrics in metrics_dict.items():

                # get best metrics after averaging over ranks

                metrics = torch.asarray(metrics, device=self.rank)

                if not self.ddp and is_multiproc():

                    # get all ranks' metrics and take the mean
                    _, metrics = all_reduce(metrics,
                                            op=torch.distributed.ReduceOp.AVG,
                                            async_op=False)

                if name == 'acc':
                    best_idx = torch.argmax(metrics)
                else:
                    best_idx = torch.argmin(metrics)

                log_dict.update({
                    f'hparams {what}/best validation {name} {what}': whats[best_idx],
                    f'hparams {what}/best validation {name}': metrics[best_idx],
                })

        if self.logs:
            self.writer.add_hparams(self.hparams, log_dict)


    def _log_metrics(self):
        self._log_separate()
        self._log_separate('step')

        self._log_together()
        self._log_together('step')


    def save(self,
             save_path=None, # only if save_dir is None
             save_dir=None, # only if save_path is None
             **kwargs):

        # TODO: deal with this when ddp for model replicas is implemented
        if self.ddp and self.rank != 0:
            return

        assert save_path is None or save_dir is None, \
            'save_path and save_dir both given which is ambiguous'

        if save_path is None:

            if save_dir is None:
                save_dir = self.save_dir

            if not os.path.exists(save_dir):
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

            if not self.ddp and is_multiproc():
                train_id = self.train_id + f'_rank{self.rank}'
            else:
                train_id = self.train_id

            save_path = os.path.join(save_dir, f'{train_id}.pt')

        if self.prints:
            print(f'Saving final model to {save_path}')

        assert self.final_model is not None, 'Call Trainer.train before attempting to save'

        model = self.final_model.module if self.ddp else self.final_model

        save_dict = {
            'steps': self.step,
            'epochs': self.epoch,
            'epochs_sgd': self.epochs_sgd,
            'epochs_codist': self.epochs_codist,
            'epochs_swa': self.epochs_swa,
            'step_valid_metrics': self.step_valid_metrics,
            'epoch_valid_metrics': self.step_valid_metrics,
            'step_train_metrics': self.step_train_metrics,
            'epoch_train_metrics': self.step_train_metrics,
            'hparams': self.hparams,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.swa_scheduler is not None:
            save_dict['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict()

        torch.save(save_dict, save_path, **kwargs)
