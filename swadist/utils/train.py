"""Training utilities
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

import os
import warnings

from copy import deepcopy
from datetime import datetime

import torch
import numpy as np

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.distributed as dist

from .losses import accuracy, CodistillationLoss
from .scheduler import LinearPolyLR
from .viz import show_imgs


class Trainer():
    """
    Class to handle training of SWADist.

    Parameters
    ----------
    model: torch.nn.Module

    train_loader: torch.utils.data.DataLoader

    valid_loader: torch.utils.data.DataLoader

    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]

    optimizer: torch.optim.Optimizer

    scheduler: torch.optim.lr_scheduler._LRScheduler

    name: str
        Name for this Trainer.
    device: torch.device, optional
        Device on which to run the code.
    world_size: int
        The number of distributed workers. If greater than 1, then .
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    log: bool
        If True, write metrics and plots to a `torch.utils.tensorboard.SummaryWriter`.
    log_dir: str
        Directory to use for SummaryWriter output and saving the model.
    save: bool
        If True, save the final model, optimizer, and scheduler states along with hyperparameters.
    save_dir: str
        Directory to use for SummaryWriter output and saving the model.
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
                 rank=0,
                 device='cpu',
                 world_size=1,
                 # logging / saving params
                 name='trainer',
                 log=False,
                 log_dir='./runs',
                 save_dir='./state_dicts',
                 n_print=1,
                 n_plot=0):

        self.device = device
        self.model = model.to(self.device)
        self.swa_model = None
        self.final_model = None
        self.ddp = hasattr(self.model, 'module')

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
        validate_per_epoch: int, optional
            Number of model evaluations on the validation data per epoch. There is always at least
            one validation at the end of each epoch.
        stopping_acc: float, optional
            Validation accuracy at which to stop training.
        """
        if epochs_codist > 0:
            assert not self.ddp, \
                'Codistillation is incompatible with DistributedDataParallel models.'
            assert self.world_size > 1, \
                "When using codistillation, world_size should be greater than 1."

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

        self.train_loss = np.inf
        self.valid_loss = np.inf
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

        self.step = 0
        self.epoch = 0

        # save hyperparameters
        hparams_dict = {
            'world_size': self.world_size,
            'batch_size': self.train_loader.batch_size,
            'epochs_sgd': epochs_sgd,
            'epochs_codist': epochs_codist,
            'epochs_swa': epochs_swa,
            'optimizer': type(self.optimizer).__name__
        }

        # add the rest of the hyperparams
        hparams_dict.update(self.optimizer.defaults)

        if self.scheduler is not None:
            hparams_dict['scheduler'] = type(self.scheduler).__name__

            if isinstance(self.scheduler, LinearPolyLR):
                hparams_dict['alpha'] = self.scheduler.alpha
                hparams_dict['decay_epochs'] = self.scheduler.alpha

        if self.swa_scheduler is not None:
            hparams_dict['scheduler'] = type(self.scheduler).__name__

            if isinstance(self.swa_scheduler, torch.optim.swa_utils.SWALR):
                hparams_dict['swa_lr'] = self.swa_scheduler.optimizer.param_groups[0]['swa_lr']
                hparams_dict['swa_anneal_epochs'] = self.swa_scheduler.anneal_epochs
                hparams_dict['swa_anneal_func'] = self.swa_scheduler.anneal_func.__name__

        self.hparams_dict = hparams_dict

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_id = f'{self.name}_{timestamp}'

        # setup logging
        if self.logs:
            # create the writer
            self.writer = SummaryWriter(f'{self.log_dir}/{train_id}')

        total_epochs = epochs_sgd + epochs_codist + epochs_swa
        if not self.world_size > 1:
            print(f'Starting {total_epochs}-epoch training loop...\n')
        else:
            print(f'Worker {self.rank+1}/{self.world_size} starting '
                  f'{total_epochs}-epoch training loop...\n')

        if self.prints:
            print(f'SGD epochs: {epochs_sgd} | '
                  f'Codistillation epochs: {epochs_codist} | '
                  f'SWA epochs: {epochs_swa}')
            print(f'DistributedDataParallel: {self.ddp}')
            print(f'Stopping accuracy: {self.stopping_acc}\n')

        # vanilla SGD / burn-in
        if epochs_sgd > 0:
            self._burn_in(epochs_sgd)

        # codistillation
        if not self.stop_early and epochs_codist > 0:
            self._codist(epochs_codist)

        # stochastic weight averaging
        if not self.stop_early and epochs_swa > 0:
            self._swa(epochs_swa)

        if not self.stop_early:
            # cache the trained network
            self.final_model = self.model

        self.model = None

        if save:
            if save_dir is None:
                save_dir = self.save_dir
            if self.world_size > 1:
                train_id = train_id + f'rank{self.rank}'
            self.save(f=os.path.join(save_dir, f'{train_id}.pt'))

        if self.logs:
            self._log_hparam_metrics()
            self.writer.close()


    def _check_for_early_stop(self, valid_acc):
        if self.stopping_acc is not None and valid_acc >= self.stopping_acc:

            # in distributed mode, check if all ranks have reached stopping acc
            if self.world_size > 1 and dist.is_initialized():
                stop_early = [torch.Tensor([rank == self.rank]).to(self.device)
                              for rank in range(self.world_size)]
                dist.all_gather(stop_early, stop_early[self.rank])
                stop_early = [t.cpu().item() for t in stop_early]
                self.stop_early = np.all(stop_early)

            if self.stop_early:
                if self.prints:
                    print('Validation accuracy target reached after {self.step} steps. '
                          'Saving current model and stopping after epoch {self.epoch}.')
                    self.prints = False
                self.final_model = deepcopy(self.model)


    def _burn_in(self, epochs):
        # epoch loop
        for _ in range(epochs):

            if self.stop_early:
                break

            self._train_epoch()


    def _codist(self, epochs):
        self.in_codist = True

        if self.prints:
            print(f'Starting codistillation phase...\n')

        # switch to codistillation loss
        self.codist_loss_fn = CodistillationLoss(self.loss_fn,
                                                 self.model,
                                                 self.device,
                                                 self.rank,
                                                 self.world_size)

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
        self.swa_model = AveragedModel(self.model)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):

            if self.stop_early:
                break

            self._train_epoch()

        # update bn statistics at end of training
        update_bn(self.train_loader, self.swa_model, self.device)


    def _train_epoch(self):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        self.model.train()

        step = self.n_batches * self.epoch
        self.epoch += 1
        loss = 0.
        acc = 0.

        if self.ddp:
            # required so that shuffling changes across epochs when using
            # DistributedSampler
            self.train_loader.sampler.set_epoch(self.epoch)

        # training loop
        for batch_idx, (image, target) in enumerate(self.train_loader):
            step += 1

            image, target = image.to(self.device), target.to(self.device)

            step_loss, step_acc = self._train_step(image, target)

            loss += step_loss
            acc += step_acc

            step_train_loss = loss / (batch_idx + 1)
            step_train_acc = acc / (batch_idx + 1)

            if self.prints and (
                    batch_idx % self.n_print == 0 or
                    batch_idx == self.n_batches - 1
            ):
                end = '' if batch_idx < self.n_batches - 1 else '\n'
                print(
                    f'\rTrain epoch: {self.epoch} -- '
                    f'Accuracy: {step_train_acc:.6f} -- '
                    f'Avg. loss ({self.loss_fn.__name__}): {step_train_loss:.6f} -- '
                    f'Batch: {batch_idx + 1}/{self.n_batches} '
                    f'({100. * (batch_idx + 1) / self.n_batches:.0f}%) -- '
                    f'Total steps: {step}',
                    end=end
                )

            # unless target acc is reached, validate steps
            if (self.val_freq and
                not self.stop_early and
                batch_idx < self.n_batches - 1 and
                (batch_idx + 1) % self.val_freq == 0 and
                 self.val_freq <= self.n_batches - (batch_idx + 1)):

                # save metrics, validate and log metrics every val_freq steps
                self.step = step
                self.step_train_loss = step_train_loss
                self.step_train_acc = step_train_acc
                self._validate(step_type='step')

        if self.n_plot > 0 and self.epoch % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.train_loader, n_imgs=2, save_idxs=True,
                                             epoch=self.epoch, log=self.logs)

        # update step training metrics at end of epoch
        if not self.stop_early:
            self.step = step
            self.step_train_loss = step_train_loss
            self.step_train_acc = step_train_acc

        # calculate epoch training metrics
        self.train_loss = loss / self.n_batches
        self.train_acc = acc / self.n_batches
        self.train_losses.append(self.train_loss)
        self.train_accs.append(self.train_acc)

        # validate the epoch, which logs metrics
        self._validate()

        if self.in_swa and self.swa_scheduler:
            # update running average of parameters
            self.swa_scheduler.step()
        elif self.scheduler:
            self.scheduler.step()


    def _train_step(self, image, target):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        image: torch.Tensor
            A batch of input images. Shape : (batch_size, channel, height, width).
        image: torch.Tensor
            A batch of corresponding labels.
        """
        # clear gradients
        self.model.zero_grad()

        # calculate the loss & gradients
        output = self.model(image)

        loss = self.loss_fn(output, target)
        loss_ = loss.clone().detach()

        if self.in_codist:
            loss += self.codist_loss_fn(image, output)

        loss.backward()

        self.optimizer.step()

        # calculate accuracy
        acc = accuracy(output, target)

        return loss_.item(), acc.item()


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

        if epoch_val and self.in_swa:
            # update the AveragedModel at the end of every epoch
            self.swa_model.update_parameters(self.model)

        self.model.eval()

        # validation loop
        with torch.inference_mode():
            for batch_idx, (image, target) in enumerate(self.valid_loader):
                image, target = image.to(self.device), target.to(self.device)

                if self.in_swa:
                    output = self.swa_model(image)
                else:
                    output = self.model(image)

                loss += self.loss_fn(output, target).item()

                acc += accuracy(output, target).item()
                step_valid_loss = loss / (batch_idx + 1)
                step_valid_acc = acc / (batch_idx + 1)

                if epoch_val and self.prints and (
                    batch_idx % self.n_print == 0 or
                    batch_idx == self.n_batches - 1
                ):
                    end = '' if batch_idx < len(self.valid_loader) - 1 else '\n\n'
                    print(
                        f'\rValidation accuracy: {step_valid_acc:.6f} -- '
                        f'Avg. loss ({self.loss_fn.__name__}): {step_valid_loss:.6f} -- '
                        f'Batch: {batch_idx + 1}/{len(self.valid_loader)} '
                        f'({100.*(batch_idx + 1) / len(self.valid_loader):.0f}%)',
                        end=end
                    )

        if epoch_val:
            # save epoch loss and acc
            self.valid_loss = step_valid_loss
            self.valid_acc = step_valid_acc

            # add to epoch history
            self.valid_losses.append(step_valid_loss)
            self.valid_accs.append(step_valid_acc)

            # log training and validation metrics for epoch
            self._log_together(step_type='epoch')
            self._log_separate(step_type='epoch')

        # save step loss (including for epoch val) if target acc has not been reached
        if not self.stop_early:
            self.step_valid_loss = step_valid_loss
            self.step_valid_acc = step_valid_acc

            # log training and validation metrics for step
            self._log_together(step_type='step')
            self._log_separate(step_type='step')

        # check if we should stop early
        self._check_for_early_stop(step_valid_acc)

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


    def save(self, f=None, **kwargs):

        assert self.final_model is not None, 'Call Trainer.train before attempting to save'

        if f is None:
            f = os.path.join(self.save_dir,
                             f'{self.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')

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

        torch.save(save_dict, f, **kwargs)
