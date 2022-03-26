"""Training utilities
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

from datetime import datetime

import torch
import numpy as np

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .losses import accuracy, CodistillationLoss
from .scheduler import LinearPolyLR
from .viz import show_imgs


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: torch.nn.Module

    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]

    optimizer: torch.optim.Optimizer

    scheduler: torch.optim.lr_scheduler._LRScheduler

    name: str
        Name for this Trainer.
    device: torch.device, optional
        Device on which to run the code.
    data_parallel: bool
        If True, wrap the model in torch.nn.parallel.DistributedDataParallel.
    world_size: int
        The number of distributed workers.
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    log: bool
        If True, write metrics and plots to a torch.utils.tensorboard.SummaryWriter.
    log_dir: str
        Directory to use for TensorBoard runs.
    log_ranks: List[int]
        A list of rank indices on which to log. If None (default) and `log` is True, logs the
        current rank.
    n_print: int
        How often to print training / validation metrics, in number of epochs.
    n_plot: int
        How often to plot or log training / validation images, in number of epochs.

    """
    def __init__(self,
                 # training params
                 model,
                 loss_fn=None,
                 optimizer=None,
                 scheduler=None,
                 # compute params
                 rank=0,
                 device='cpu',
                 world_size=1,
                 data_parallel=False,
                 # logging params
                 name='trainer',
                 log=False,
                 log_dir='./runs',
                 log_ranks=None,
                 n_print=1,
                 n_plot=0):

        self.device = device
        self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_print = n_print
        self.n_plot = n_plot

        self.rank = rank
        self.world_size = world_size
        self.data_parallel = data_parallel
        if self.data_parallel:
            self.model = DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )

        self.name = name
        if log_ranks is None:
            log_ranks = [self.rank]

            self.log = log and self.rank in log_ranks
        self.log_dir = log_dir


    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)


    def train(self,
              train_loader,
              valid_loader,
              epochs=10,
              epochs_codist=0,
              epochs_swa=0,
              validations_per_epoch=None,
              stopping_acc=np.inf):
        """
        Trains the model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
        valid_loader: torch.utils.data.DataLoader, optional
        epochs: int, optional
            Number of epochs to train the model (burn-in).
        epochs_swa: int, optional
            Number of additional epochs for stochastic weight averaging.
        codistill: bool, optional
            If True, use codistillation.
        stopping_acc: float, optional
            Validation accuracy at which to stop training.
        """
        self.in_codist = False
        self.in_swa = False

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_batches = len(self.train_loader)

        self.val_freq = None

        if validations_per_epoch and min(self.n_batches, validations_per_epoch) - 1 > 0:

            self.val_freq = int(np.floor(self.n_batches / validations_per_epoch))

        self.stopping_acc = stopping_acc
        self.stop_early = False

        self.train_loss = np.inf
        self.valid_loss = np.inf
        self.train_acc = 0.0
        self.valid_acc = 0.0

        self.step_train_loss = np.inf
        self.step_valid_loss = np.inf
        self.step_train_acc = 0.0
        self.step_valid_acc = 0.0

        self.step = 0
        self.epoch = 0

        if self.log and self.rank == 0:

            # setup logging

            hparam_dict = {
                'batch_size': self.train_loader.batch_size,
                'epochs': epochs,
                'epochs_codist': epochs_codist,
                'epochs_swa': epochs_swa
            }

            # simplified hparams for filename
            hparam_str = '_'.join([f'{k}={v}' for k, v in hparam_dict.items()])

            # add the rest of the hyperparams
            hparam_dict['optimizer'] = type(self.optimizer).__name__
            hparam_dict.update(self.optimizer.defaults)

            if self.scheduler is not None:
                hparam_dict['scheduler'] = type(self.scheduler).__name__

            if isinstance(self.scheduler, LinearPolyLR):
                hparam_dict['alpha'] = self.scheduler.alpha
                hparam_dict['decay_epochs'] = self.scheduler.alpha

            self.hparam_dict = hparam_dict

            # create the writer
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(
                f'{self.log_dir}/{self.name}_{timestamp}_rank={self.rank}_{hparam_str}'
            )

        max_epochs = epochs + epochs_codist + epochs_swa
        if not self.data_parallel:
            print(f'Starting {max_epochs}-epoch training loop...\n')
        else:
            print(f'Worker {self.rank+1}/{self.world_size} starting '
                  f'{max_epochs}-epoch training loop...\n')

        self.prints = self.n_print > 0 and self.rank == 0

        # vanilla SGD / burn-in
        if epochs > 0:
            self._burn_in(epochs)

        # codistillation
        if not self.stop_early and epochs_codist > 0:
            self._codist(epochs_codist)

        # stochastic weight averaging
        if not self.stop_early and epochs_swa > 0:
            self._swa(epochs_swa)

        if self.log:
            self.writer.close()


    def _check_for_early_stop(self):
        if self.valid_acc >= self.stopping_acc:
            print('Validation accuracy target reached. Stopping early after '
                  f'{self.epoch} epochs ({self.step} steps)')
            self.stop_early = True


    def _burn_in(self, epochs):
        # epoch loop
        for _ in range(epochs):

            if self.stop_early:
                break

            self._train_epoch()

        self._log_hparam_metrics()


    def _codist(self, epochs):
        self.in_codist = True

        # remove DDP
        self.model = self.model.module.to(self.device)

        # switch to codistillation loss
        self.loss_fn = CodistillationLoss(self.loss_fn, self.model,
                                          self.device, self.rank, self.world_size)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):

            if self.stop_early:
                break

            self._train_epoch()

        self._log_hparam_metrics()


    def _swa(self, epochs):
        self.in_swa = True

        # save the model at end of first phase and create the SWA model
        self.base_model = self.model
        self.model = AveragedModel(self.base_model).to(self.device)

        # epoch loop
        for _ in range(self.epoch, self.epoch + epochs):
            # update running average of parameters
            self.model.update_parameters(self.model)

            if self.stop_early:
                break

            self._train_epoch()

        self._log_hparam_metrics()


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

        step = self.n_batches*self.epoch
        self.epoch += 1
        loss = 0.
        acc = 0.

        # training loop
        for batch_idx, (image, target) in enumerate(self.train_loader):
            step += 1

            image, target = image.to(self.device), target.to(self.device)

            step_loss, step_acc = self._train_step(image, target)

            loss += step_loss
            acc += step_acc

            step_train_loss = loss / (batch_idx + 1)
            step_train_acc = acc / (batch_idx + 1)

            if self.prints and self.epoch % self.n_print == 0:
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
            if not self.stop_early and self.val_freq and \
               (batch_idx + 1) % self.val_freq == 0 and batch_idx < self.n_batches - 1:

                # save metrics, validate and log metrics every val_freq steps
                self.step = step
                self.step_train_loss = step_train_loss
                self.step_train_acc = step_train_acc
                self._validate(step_type='step')


        if self.n_plot > 0 and self.epoch % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.train_loader, n_imgs=2, save_idxs=True,
                                             epoch=self.epoch, log=self.log)

        # calculate epoch training metrics
        self.train_loss = loss / self.n_batches
        self.train_acc = acc / self.n_batches

        if self.in_swa:
            # update bn statistics before validation
            torch.optim.swa_utils.update_bn(self.train_loader, self.model)

        # validate the epoch, which logs metrics
        self._validate()

        if self.scheduler:
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
        if self.in_codist:
            loss = self.loss_fn(image, output, target)
        else:
            loss = self.loss_fn(output, target)
        loss.backward()

        # update the optimizer step
        self.optimizer.step()

        # calculate accuracy
        acc = accuracy(output, target)

        return loss.item(), acc.item()


    def _validate(self, step_type='epoch'):
        """
        Validates the current model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch_val: bool
            If True, this validation is at the end of an epoch. Otherwise it's a step validation.

        Return
        ------
        mean_valid_loss: float
        """
        self.model.eval()
        loss = 0.
        acc = 0.
        epoch_val = step_type == 'epoch'
        step = self.epoch if epoch_val else self.step

        # validation loop
        for batch_idx, (image, target) in enumerate(self.valid_loader):
            with torch.inference_mode():
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)
                if self.in_codist:
                    loss += self.loss_fn(image, output, target).item()
                else:
                    loss += self.loss_fn(output, target).item()
                acc += accuracy(output, target).item()
                step_valid_loss = loss / (batch_idx + 1)
                step_valid_acc = acc / (batch_idx + 1)

                if epoch_val and self.prints and step % self.n_print == 0:
                    end = '' if batch_idx < len(self.valid_loader) - 1 else '\n\n'
                    print(
                        f'\rValidation accuracy: {step_valid_acc:.6f} -- '
                        f'Avg. loss ({self.loss_fn.__name__}): {step_valid_loss:.6f} -- '
                        f'Batch: {batch_idx + 1}/{len(self.valid_loader)} '
                        f'({100.*(batch_idx + 1) / len(self.valid_loader):.0f}%)',
                        end=end
                    )

        if epoch_val:
            self.valid_loss = step_valid_loss
            self.valid_acc = step_valid_acc
        else:
            self.step_valid_loss = step_valid_loss
            self.step_valid_acc = step_valid_acc

        # check if we should stop early
        self._check_for_early_stop()

        # log training and validation metrics
        self._log_together(step_type=step_type)
        self._log_separate(step_type=step_type)

        # plot or log feature maps and filters
        if epoch_val and self.n_plot > 0 and step % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.valid_loader, n_imgs=2, save_idxs=True,
                                             epoch=step, stage='validation', log=self.log)
            _ = self.plot_or_log_filters(save_idxs=True, epoch=step, log=self.log)


    def _log_separate(self, step_type='epoch'):
        """Separately log training and validation metrics.
        """
        if self.log:
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
        if self.log:
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
        # TODO: move this to experiments.py
        if self.log:
            hparam_dict = self.hparam_dict.copy()
            hparam_dict['epochs'] = self.epoch
            hparam_dict['steps'] = self.step
            self.writer.add_hparams(
                hparam_dict,
                {'hparams epoch/valid acc': self.train_acc,
                 'hparams epoch/valid loss': self.train_loss,
                 'hparams step/valid acc': self.step_valid_acc,
                 'hparams step/valid loss': self.step_valid_loss,
                 'hparams epoch/train acc': self.train_acc,
                 'hparams epoch/train loss': self.train_loss,
                 'hparams step/train acc': self.step_valid_acc,
                 'hparams step/train loss': self.step_valid_loss}
            )


    def plot_or_log_activations(self, loader, img_idx_dict=None, n_imgs=None, save_idxs=False,
                                epoch=None, stage='train', random=False, log=False):
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

            if log and self.rank == 0:
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
                            epoch=None, random=False, log=False):
        self.model.eval()

        if w_idx_dict is None:
            if not random and hasattr(self, 'filter_idx_dict'):
                w_idx_dict = self.filter_idx_dict
            filters, w_idx_dict = self.model.get_filters(w_idx_dict)
        else:
            filters, w_idx_dict = self.model.get_filters()

        if log and self.rank == 0:
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
