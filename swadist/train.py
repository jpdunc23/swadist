"""Trainer
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

import os
import warnings

from copy import deepcopy
from datetime import datetime

import torch
import numpy as np

from torch.nn.utils import parameters_to_vector
from torchvision.utils import make_grid
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.tensorboard import SummaryWriter

from .viz import show_imgs
from .optim import LinearPolyLR
from .metrics import accuracy, CodistillationLoss
from .distributed import all_gather, is_multiproc


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
    n_plot: int
        How often to plot or log training / validation images, in number of epochs.

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
                 n_print=10,
                 n_plot=0):

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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_scheduler = swa_scheduler

        self.rank = rank
        self.world_size = world_size

        self.n_print = n_print
        self.n_plot = n_plot
        self.prints = self.n_print > 0 and self.rank == 0

        self.name = name
        self.logs = log and self.rank == 0
        self.log_dir = log_dir

        self.save_dir = save_dir
        self.train_id = None

        # set during training
        self.hparams_dict = None
        self.in_codist = None
        self.in_swa = None
        self.val_freq = None
        self.stopping_acc = None
        self.stop_early = None
        self.train_loss = None
        self.valid_loss = None
        self.train_acc = None
        self.valid_acc = None
        self.train_losses = None
        self.valid_losses = None
        self.train_accs = None
        self.valid_accs = None
        self.step_train_loss = None
        self.step_valid_loss = None
        self.step_train_acc = None
        self.step_valid_acc = None
        self.step = None
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
              validate_per_epoch: int=None,
              stopping_acc: float=None,
              save: bool=False,
              save_dir: str=None):
        """
        Trains the model.

        Parameters
        ----------
        epochs_sgd: int, optional
            Number of epochs to train the model using vanilla SGD.
        epochs_codist: bool, optional
            Number of epochs using codistillation.
        epochs_swa: int, optional
            Number of epochs using stochastic weight averaging.
        swadist: bool, optional
            If True, use codistillation during the SWA phase. (NOT IMPLEMENTED)
        codist_kwargs: dict, optional
            Additional keyword arguments to pass to `CodistillationLoss` initializer.
        validate_per_epoch: int, optional
            Number of model evaluations on the validation data per epoch. There is always at least
            one validation at the end of each epoch.
        stopping_acc: float, optional
            Validation accuracy at which to stop training.
        save: bool
            If True, save the final model, optimizer, and scheduler states and hyperparameters.
        save_dir: str
            Directory to use for saving the model.

        """
        # save hyperparameters
        hparams_dict = {
            'world_size': self.world_size,
            'batch_size': self.train_loader.batch_size,
            'epochs_sgd': epochs_sgd,
            'epochs_codist': epochs_codist,
            'epochs_swa': epochs_swa,
            'optimizer': type(self.optimizer).__name__,
            'loss_fn': self.loss_fn.__name__,
        }

        if epochs_codist > 0:
            assert not self.ddp, \
                'Codistillation is incompatible with DistributedDataParallel models.'
            assert is_multiproc(), \
                ('To run codistillation, use multiprocessing with world_size greater than 1 ',
                 '(see swadist.utils.spawn_fn).')

            if codist_kwargs is None:
                codist_kwargs = {}

            hparams_dict['codist_sync_freq'] = codist_kwargs.get('sync_freq', 50)
            hparams_dict['codist_loss_fn'] = codist_kwargs.get('loss_fn', self.loss_fn).__name__

        self.in_codist = False
        self.in_swa = False

        if validate_per_epoch:
            n_valid = min(self.n_batches, validate_per_epoch)
            if n_valid < validate_per_epoch:
                warnings.warn(f'Requested {validate_per_epoch} validations per epoch but only '
                              f'have {n_valid} batches. Using validate_per_epoch={n_valid}.')
            self.val_freq = self.n_batches // n_valid

        self.stopping_acc = stopping_acc
        self.stop_early = False

        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

        self.step = 0
        self.epoch = 0

        # add the rest of the hyperparams
        hparams_dict.update(self.optimizer.defaults)

        if self.scheduler is not None:
            hparams_dict['scheduler'] = type(self.scheduler).__name__

            if isinstance(self.scheduler, LinearPolyLR):
                hparams_dict['scheduler_alpha'] = self.scheduler.alpha
                hparams_dict['scheduler_decay_epochs'] = self.scheduler.alpha

        if epochs_swa > 0 and self.swa_scheduler is not None:

            if isinstance(self.swa_scheduler, dict):
                hparams_dict['swa_scheduler'] = 'SWALR'
                for key, val in self.swa_scheduler.items():
                    hparams_dict[f'swa_scheduler_{key}'] = val

            else:
                hparams_dict['swa_scheduler'] = type(self.swa_scheduler).__name__
                if isinstance(self.swa_scheduler, SWALR):
                    hparams_dict['swa_scheduler_swa_lr'] = self.swa_scheduler.optimizer.param_groups[0]['swa_lr']
                    hparams_dict['swa_scheduler_anneal_epochs'] = self.swa_scheduler.anneal_epochs
                    hparams_dict['swa_scheduler_anneal_strategy'] = self.swa_scheduler.anneal_func.__name__

        self.hparams_dict = hparams_dict

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.train_id = f'{self.name}_{timestamp}'

        # setup logging
        if self.logs:
            # create the writer
            self.writer = SummaryWriter(f'{self.log_dir}/{train_id}')

        total_epochs = epochs_sgd + epochs_codist + epochs_swa
        if not is_multiproc():
            print(f'Starting {total_epochs}-epoch training loop...')
            print(f'Random seed: {torch.seed()}\n')
        else:
            print(f'Worker {self.rank+1}/{self.world_size} starting '
                  f'{total_epochs}-epoch training loop...')

        if self.prints:
            print(f'SGD epochs: {epochs_sgd} | '
                  f'Codistillation epochs: {epochs_codist} | '
                  f'SWA epochs: {epochs_swa}')
            print(f'DistributedDataParallel: {self.ddp}')
            print(f'Stopping accuracy: {self.stopping_acc}\n')
            # print('Param preview:')
            # print(next(model.parameters())[0], '\n')

        # vanilla SGD / burn-in
        if epochs_sgd > 0:
            self._sgd(epochs_sgd)

        # codistillation
        if not self.stop_early and epochs_codist > 0:
            self._codist(epochs_codist, codist_kwargs)

        # stochastic weight averaging
        if not self.stop_early and epochs_swa > 0:
            self._swa(epochs_swa)

        if not self.stop_early:
            # cache the trained network
            self.final_model = self.model

        self.model = None

        if save:
            self.save(save_dir=save_dir)

        if self.logs:
            self._log_hparam_metrics()
            self.writer.close()


    def _check_for_early_stop(self, valid_acc):
        if self.stopping_acc is not None and valid_acc >= self.stopping_acc:

            # in distributed mode, check if all ranks have reached stopping acc
            if is_multiproc():
                _, stop_early = all_gather(torch.asarray(True, device=self.device),
                                           rank=self.rank,
                                           world_size=self.world_size)
                self.stop_early = parameters_to_vector(stop_early).all().cpu().item()

            if self.stop_early:
                if self.prints:
                    print(f'Validation accuracy target reached after {self.step} steps. '
                          f'Caching current model and stopping after epoch {self.epoch}.')
                    self.prints = False
                self.final_model = deepcopy(self.model.cpu())
                self.model.to(self.device)


    def _sgd(self, epochs):
        # epoch loop
        for _ in range(epochs):

            if self.stop_early:
                break

            self._train_epoch()


    def _codist(self, epochs, codist_kwargs):
        self.in_codist = True

        if self.prints:
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

            self._train_epoch()


    def _swa(self, epochs):
        self.in_swa = True

        if self.prints:
            print(f'Starting SWA phase...\n')

        # create the model for SWA
        # TODO: may want to store other model on cpu when not in use
        self.swa_model = AveragedModel(self.model)

        if isinstance(self.swa_scheduler, dict):
            self.swa_scheduler = SWALR(self.optimizer, **self.swa_scheduler)

        # if self.swalr_kwargs is not None:
        #     self.swa_scheduler = SWALR(self.optimizer, **self.swalr_kwargs)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):

            if self.stop_early:
                break

            self._train_epoch()


    def _train_epoch(self):
        """Trains the model for one epoch.

        """
        self.model.train()

        step = self.n_batches * self.epoch
        self.epoch += 1
        loss = 0.
        acc = 0.
        codist_loss = 0.
        metrics = {}

        if self.ddp:
            # required so that shuffling changes across epochs when using
            # DistributedSampler
            self.train_loader.sampler.set_epoch(self.epoch)

        # training loop
        for batch_idx, (x, y) in enumerate(self.train_loader):
            step += 1

            x, y = x.to(self.device), y.to(self.device)

            # mean over the batch size
            step_metrics = self._train_step(x, y)

            for name, metric in step_metrics.items():
                # running sum of batch mean metrics
                metrics[name] = metrics.get(name, 0.) + metric

            train_end = batch_idx == self.n_batches - 1

            if train_end:
                for name, metric in step_metrics.items():
                    # running sum over steps in epoch
                    metrics[name] = metrics[name] / self.n_batches

            if self.prints and (batch_idx % self.n_print == 0 or train_end):

                metrics_ = metrics if train_end else step_metrics
                metrics_str = 'Metrics (epoch mean): ' if train_end else 'Metrics (batch mean): '
                metrics_str = metrics_str + \
                    f'{self.loss_fn.__name__}={metrics_["loss"]:.6f} <> '

                metric_strs = []
                for name, metric in metrics_.items():
                    if name != 'loss':
                        metric_strs.append(f'{name}={metric:.6f}')

                metrics_str += ' <> '.join(metric_strs)

                end = '\n' if train_end else ''
                print(
                    f'\rTrain epoch: {self.epoch} | {metrics_str} | '
                    f'Batch (size {x.shape[0]}): {batch_idx + 1}/{self.n_batches} '
                    f'({100. * (batch_idx + 1) / self.n_batches:.0f}%) | '
                    f'Total steps: {step}',
                    end=end
                )

            # TODO: replace (step_)loss and (step_)acc with metrics

            # if there are any validation steps within the epoch and target acc
            # hasn't been reached, validate steps
            if (self.val_freq and
                not train_end and
                not self.stop_early and
                (batch_idx + 1) % self.val_freq == 0 and
                 self.val_freq <= self.n_batches - (batch_idx + 1)):

                # save metrics, validate and log metrics every val_freq steps
                self.step = step
                self.step_train_loss = step_metrics['loss']
                self.step_train_acc = step_metrics['acc']
                self._validate(step_type='step')

        if self.n_plot > 0 and self.epoch % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.train_loader, n_imgs=2, save_idxs=True,
                                             epoch=self.epoch, log=self.logs)

        # update step training metrics at end of epoch
        if not self.stop_early:
            self.step = step
            self.step_train_loss = step_metrics['loss']
            self.step_train_acc = step_metrics['acc']

        # calculate epoch training metrics
        self.train_loss = metrics['loss']
        self.train_acc = metrics['acc']
        self.train_losses.append(self.train_loss)
        self.train_accs.append(self.train_acc)

        # validate the epoch, which logs metrics
        self._validate()

        if self.in_swa and self.swa_scheduler is not None:
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
            'loss': self.loss_fn(output, y),
            'acc': accuracy(output, y),
        }

        if self.in_codist:
            metrics['codist_loss'] = self.codist_loss_fn(x, output)
            codist_loss = metrics['codist_loss']
            # codist_steps = self.codist_loss_fn.step
            # if codist_steps < 1750:
                # codist_loss = codist_steps * codist_loss / 1750
            (metrics['loss'] + codist_loss).backward()
        else:
            metrics['loss'].backward()

        self.optimizer.step()

        for name, metric in metrics.items():
            metrics[name] = metric.item()

        return metrics


    def _validate(self, step_type='epoch'):
        """
        Validates the current model.

        Parameters
        ----------
        step_type: str
            Either 'epoch' or 'step'.

        """

        loss = 0.
        acc = 0.
        epoch_val = step_type == 'epoch'
        step = self.epoch if epoch_val else self.step
        prints = epoch_val and self.prints
        n_batches = len(self.valid_loader)

        if self.in_swa:
            # update the AveragedModel and batchnorm stats before every eval
            self.swa_model.update_parameters(self.model)
            update_bn(self.train_loader, self.swa_model, self.device)

        self.model.eval()

        # validation loop
        with torch.inference_mode():
            for batch_idx, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                if self.in_swa:
                    output = self.swa_model(x)
                else:
                    output = self.model(x)

                batch_loss = self.loss_fn(output, y).item()
                loss += batch_loss

                batch_acc = accuracy(output, y).item()
                acc += batch_acc

                # final validation batch?
                val_end = batch_idx == (n_batches - 1)

                if val_end:
                    loss = loss / n_batches
                    acc = acc / n_batches
                    batch_loss = loss
                    batch_acc = acc

                metrics = [('\r', batch_acc, batch_loss)]

                if self.n_print > 0 and (val_end or (batch_idx % self.n_print) == 0):

                    end = '\n' if val_end else ''

                    if val_end and is_multiproc():

                        if prints:
                            print('\r', end='')

                        # gather all ranks' metrics at end of validation
                        _, metrics = all_gather(torch.asarray(metrics[0][1:], device=self.device),
                                                rank=self.rank,
                                                world_size=self.world_size)

                        for i, arr in enumerate(metrics):
                            arr.cpu()
                            metrics[i] = (f'Rank {i} | ',
                                          arr[0].item(), # rank i acc
                                          arr[1].item()) # rank i loss

                    if prints:
                        # only print on rank 0
                        for (begin, batch_acc, batch_loss) in metrics:
                            print(f'{begin}Validation | '
                                  f'{self.loss_fn.__name__}={batch_loss:.6f} <> '
                                  f'accuracy={batch_acc:.6f} | '
                                  f'Batch: {batch_idx + 1}/{n_batches} '
                                  f'({100.*(batch_idx + 1) / n_batches:.0f}%)',
                                  end=end)

        if prints:
            print()

        if epoch_val:
            # save epoch loss and acc
            self.valid_loss = loss
            self.valid_acc = acc

            # add to epoch history
            self.valid_losses.append(loss)
            self.valid_accs.append(acc)

            # log training and validation metrics for epoch
            self._log_together(step_type='epoch')
            self._log_separate(step_type='epoch')

        if not self.stop_early:
            # save step loss (including for epoch val) if target acc has not been reached
            self.step_valid_loss = loss
            self.step_valid_acc = acc

            # log training and validation metrics for step
            self._log_together(step_type='step')
            self._log_separate(step_type='step')

            # check if we should stop early
            self._check_for_early_stop(acc)

        # plot or log feature maps and filters
        if epoch_val and self.n_plot > 0 and step % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.valid_loader, n_imgs=2, save_idxs=True,
                                             epoch=step, stage='validation', log=self.logs)
            _ = self.plot_or_log_filters(save_idxs=True, epoch=step, log=self.logs)


    def _log_separate(self, step_type='epoch'):
        """Separately log training and validation metrics.
        """
        if self.logs:
            if step_type == 'epoch':
                train_acc = self.train_acc
                train_loss = self.train_loss
                valid_acc = self.valid_acc
                valid_loss = self.valid_loss
                step = self.epoch
            else:
                train_acc = self.step_train_acc
                train_loss = self.step_train_loss
                valid_acc = self.step_valid_acc
                valid_loss = self.step_valid_loss
                step = self.step
            if not self.stop_early:
                self.writer.add_scalar(f'{step_type} acc/valid', valid_acc, step)
                self.writer.add_scalar(f'{step_type} acc/train', train_acc, step)
                self.writer.add_scalar(f'{step_type} loss/valid', valid_loss, step)
                self.writer.add_scalar(f'{step_type} loss/train', train_loss, step)


    def _log_together(self, step_type='epoch'):
        """Log training and validation metrics together.
        """
        if self.logs:
            if step_type == 'epoch':
                train_acc = self.train_acc
                train_loss = self.train_loss
                valid_acc = self.valid_acc
                valid_loss = self.valid_loss
                step = self.epoch
            else:
                train_acc = self.step_train_acc
                train_loss = self.step_train_loss
                valid_acc = self.step_valid_acc
                valid_loss = self.step_valid_loss
                step = self.step
            if not self.stop_early:
                self.writer.add_scalars(f'{step_type} acc',
                                        {'valid': valid_acc, 'train': train_acc}, step)
                self.writer.add_scalars(f'{step_type} loss',
                                        {'valid': valid_loss, 'train': train_loss}, step)


    def _log_hparam_metrics(self):
        if self.logs:
            self.writer.add_hparams(
                self.hparams_dict,
                {'hparams epoch/epoch': self.epoch,
                 'hparams epoch/valid acc': self.valid_acc,
                 'hparams epoch/valid loss': self.valid_loss,
                 'hparams epoch/train acc': self.train_acc,
                 'hparams epoch/train loss': self.train_loss,
                 'hparams step/step': self.step,
                 'hparams step/valid acc': self.step_valid_acc,
                 'hparams step/valid loss': self.step_valid_loss,
                 'hparams step/train acc': self.step_train_acc,
                 'hparams step/train loss': self.step_train_loss}
            )


    # MAY NOT BE WORKING
    def plot_or_log_activations(self, loader, img_idx_dict=None, n_imgs=None, save_idxs=False,
                                epoch=None, stage='train', random=False):
        """
        img_idx_dict: dict
            Has entries like { 'img_idxs': [1, 2, 3], 'layer1/conv1': [10, 7, 23], ...}.
        save_idxs: bool
            If True, store the existing or new image indices in this Trainer.
        """
        self.model.eval()
        new_idx_dict = False

        if img_idx_dict is None:
            if not random and hasattr(self, f'{stage}_img_idx_dict'):
                img_idx_dict = getattr(self, f'{stage}_img_idx_dict')
                img_idxs = img_idx_dict['img_idxs']
            else:
                new_idx_dict = True
                img_idx_dict = {}
                n_imgs = 1 if n_imgs is None else n_imgs
                img_idxs = np.random.choice(len(loader.dataset), size=n_imgs, replace=False)
                img_idxs.sort()
                img_idx_dict['img_idxs'] = img_idxs
        else:
            img_idxs = img_idx_dict['img_idxs']

        for i, img_idx in enumerate(img_idxs):
            imgs = []
            image = loader.dataset[img_idx][0]
            original = loader.dataset.inv_transform(image)
            imgs.append(original)
            image = image[None, :]

            for conv in self.model.convs:
                out = self.model.partial_forward(image, conv)
                if new_idx_dict:
                    conv_idx = np.random.choice(out.shape[1], replace=False)
                    if not conv in img_idx_dict:
                        img_idx_dict[conv] = [conv_idx]
                    else:
                        img_idx_dict[conv].append(conv_idx)
                else:
                    conv_idx = img_idx_dict[conv][i]
                imgs.append(out[0, conv_idx, :, :][None, :, :])

            if self.logs:
                self.writer.add_image(f'{stage}/{img_idx}_0_original', original, epoch)
                for j, conv in enumerate(self.model.convs):
                    self.writer.add_image(f'{stage}/{img_idx}_{j+1}_{conv}', imgs[j+1], epoch)
            else:
                titles = ['original']
                titles.extend(self.model.convs)
                suptitle = f'{stage} activations'
                if epoch is not None:
                    suptitle += f' after epoch {epoch + 1}'
                show_imgs(imgs, suptitle, titles=titles)

        if save_idxs:
            setattr(self, f'{stage}_img_idx_dict', img_idx_dict)

        return img_idx_dict

    # MAY NOT BE WORKING
    def plot_or_log_filters(self, w_idx_dict=None, save_idxs=False,
                            epoch=None, random=False):
        self.model.eval()

        if w_idx_dict is None:
            if not random and hasattr(self, 'filter_idx_dict'):
                w_idx_dict = self.filter_idx_dict
            filters, w_idx_dict = self.model.get_filters(w_idx_dict)
        else:
            filters, w_idx_dict = self.model.get_filters()

        if self.logs:
            for conv, (idxs, filter_group) in filters.items():
                tag = f'{conv}/{idxs}'
                self.writer.add_images(tag, filter_group, epoch)
        else:
            imgs = [make_grid(f, nrow=2) for _, (_, f) in filters.items()]
            suptitle = 'filters'
            if epoch is not None:
                suptitle += f' after epoch {epoch + 1}'
            show_imgs(imgs, suptitle, list(w_idx_dict))

        if save_idxs:
            self.filter_idx_dict = w_idx_dict

        return w_idx_dict


    def save(self,
             save_path=None, # only if save_dir is None
             save_dir=None, # only if save_path is None
             **kwargs):

        assert save_path is None or save_dir is None, \
            'save_path and save_dir both given which is ambiguous'

        if save_path is None:

            if save_dir is None:
                save_dir = self.save_dir

            if is_multiproc():
                train_id = self.train_id + f'rank{self.rank}'

            save_path = os.path.join(save_dir, f'{train_id}.pt')

        if self.prints:
            print(f'Saving final model to {save_path}')

        assert self.final_model is not None, 'Call Trainer.train before attempting to save'

        model = self.final_model.module if self.ddp else self.final_model

        save_dict = {'epoch': self.epoch,
                     'epoch_valid_acc': self.valid_acc,
                     'epoch_valid_loss': self.valid_loss,
                     'step': self.step,
                     'step_valid_acc': self.step_valid_acc,
                     'step_valid_loss': self.step_valid_loss,
                     'hparams_dict': self.hparams_dict,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}

        if self.scheduler is not None:
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.swa_scheduler is not None:
            save_dict['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict()

        torch.save(save_dict, save_path, **kwargs)
