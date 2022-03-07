"""Training utilities
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

from datetime import datetime
import numpy as np
import torch

from torch.optim.swa_utils import AveragedModel, update_bn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .losses import accuracy, CodistillationLoss
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

    device: torch.device, optional
        Device on which to run the code.
    name: str
        Name for this Trainer.
    n_print: int
        How often to print training / validation metrics.
    n_plot: int
        How often to plot or log training / validation images.
    log: bool
        If True, write metrics and plots to a torch.utils.tensorboard.SummaryWriter.
    log_dir: str

    """
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device='cpu',
                 name='trainer', n_print=1, n_plot=0, log=False,
                 log_dir='./runs', data_parallel=False, rank=0, world_size=1):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.n_print = n_print
        self.n_plot = n_plot
        self.name = name
        self.log = log
        self.log_dir = log_dir
        self.rank = rank
        self.world_size = world_size
        self.data_parallel = data_parallel
        if self.log:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(f'{log_dir}/{name}_rank{rank}_{timestamp}')
        if self.data_parallel:
            self.model = DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )


    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)


    def train(self, train_loader, valid_loader=None,
              epochs=10, codist_epochs=0, swa_epochs=0,
              validations_per_epoch=None, stopping_acc=np.inf):
        """
        Trains the model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
        valid_loader: torch.utils.data.DataLoader, optional
        epochs: int, optional
            Number of epochs to train the model (burn-in).
        swa_epochs: int, optional
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

        self.valid_freq = None
        if self.valid_loader and validations_per_epoch:
            if min(self.n_batches, validations_per_epoch) - 1 > 0:
                self.valid_freq = int(np.floor(
                    self.n_batches / validations_per_epoch
                ))

        self.stopping_acc = stopping_acc
        self.early_stop = False

        self.total_train_epochs = epochs + codist_epochs + swa_epochs

        self.train_losses = np.empty(self.total_train_epochs)
        self.train_accs = np.empty(self.total_train_epochs)

        self.valid_losses = np.empty(self.total_train_epochs)
        self.valid_accs = np.empty(self.total_train_epochs)

        if not self.data_parallel:
            print(f'Starting {self.total_train_epochs}-epoch training loop...')
        else:
            print(f'Worker {self.rank+1}/{self.world_size} starting {self.total_train_epochs}-epoch training loop...')

        self.can_print = self.n_print > 0 and self.rank == 0

        # vanilla SGD / burn-in
        self._burn_in(epochs)

        # codistillation
        self._codist(codist_epochs)

        # stochastic weight averaging
        self._swa(swa_epochs)

        if self.log:
            self.writer.close()


    def _check_for_early_stop(self, valid_acc, epoch, step):
        if valid_acc >= self.stopping_acc:
            print(f'Validation accuracy target reached. Stopping early after {epoch + 1} epochs'
                  f' ({step} steps)')
            self.total_train_epochs = epoch + 1
            self.total_train_steps = step + 1
            self.early_stop = True


    def _burn_in(self, epochs):
        # epoch loop
        for epoch in range(epochs):

            if self.early_stop:
                break

            self._train_epoch(epoch)


    def _codist(self, epochs):
        self.in_codist = True

        # remove DDP
        self.model = self.model.module.to(self.device)

        # switch to codistillation loss
        self.loss_fn = CodistillationLoss(self.loss_fn, self.model,
                                          self.device, self.rank, self.world_size)

        # epoch loop
        for epoch in range(epochs):

            if self.early_stop:
                break

            self._train_epoch(epoch)


    def _swa(self, epochs):
        self.in_swa = True

        # save the model at end of first phase and create the SWA model
        self.base_model = self.model
        self.model = AveragedModel(self.base_model).to(self.device)

        # epoch loop
        for epoch in range(epochs):
            # update running average of parameters
            self.model.update_parameters(self.model)

            if self.early_stop:
                break

            self._train_epoch(epoch)

        # update bn stats at the end of training
        update_bn(self.train_loader, self.model)


    def _train_epoch(self, epoch):
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
        epoch_loss = 0.
        epoch_acc = 0.
        n_batches = self.n_batches
        val_freq = self.valid_freq

        # training loop
        for batch_idx, (image, target) in enumerate(self.train_loader):
            step = epoch*n_batches + batch_idx
            image, target = image.to(self.device), target.to(self.device)

            iter_loss, iter_acc = self._train_iteration(image, target)
            # self.step_loss.append(iter_loss)
            # self.step_acc.append(iter_acc)

            epoch_loss += iter_loss
            epoch_acc += iter_acc

            if self.log:
                self.writer.add_scalar('Step loss/training', iter_loss, step)
                self.writer.add_scalar('Step accuracy/training', iter_acc, step)

            if self.can_print and (epoch + 1) % self.n_print == 0:
                end = '' if batch_idx < n_batches - 1 else '\n'
                print(
                    f'\rTrain epoch: {epoch + 1} -- '
                    f'Accuracy: {epoch_acc / (batch_idx + 1):.6f} -- '
                    f'Avg. loss ({self.loss_fn.__name__}): {epoch_loss / (batch_idx + 1):.6f} -- '
                    f'Batch: {batch_idx + 1}/{n_batches} '
                    f'({100. * (batch_idx + 1) / n_batches:.0f}%) -- '
                    f'Total steps: {step + 1}',
                    end=end
                )

            # validate every valid_freq steps
            if batch_idx < n_batches - 1 and val_freq:
                if (batch_idx + 1) % val_freq == 0:
                    _, _ = self._validate(step, epoch=False)

        if self.n_plot > 0 and (epoch + 1) % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.train_loader, n_imgs=2, save_idxs=True,
                                             epoch=epoch, log=self.log)

        # calculate epoch training metrics
        mean_epoch_loss = epoch_loss / n_batches
        mean_epoch_acc = epoch_acc / n_batches
        self.train_losses[epoch] = mean_epoch_loss
        self.train_accs[epoch] = mean_epoch_acc

        if self.log:
            self.writer.add_scalar('Epoch loss/training', mean_epoch_loss, epoch)
            self.writer.add_scalar('Epoch accuracy/training', mean_epoch_acc, epoch)

        if self.valid_loader:
            if self.in_swa:
                # update bn statistics before validation
                torch.optim.swa_utils.update_bn(self.train_loader, self.model)

            # validate the epoch
            mean_valid_loss, mean_valid_acc = self._validate(global_step=epoch)
            self.valid_losses[epoch] = mean_valid_loss
            self.valid_accs[epoch] = mean_valid_acc

            # check if we should stop early
            self._check_for_early_stop(mean_valid_acc, epoch, step)

            if self.log:
                self.writer.add_scalar('Step loss/validation', mean_valid_loss, step)
                self.writer.add_scalar('Step accuracy/validation', mean_valid_acc, step)

        if self.scheduler:
            self.scheduler.step()


    def _train_iteration(self, image, target):
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


    def _validate(self, global_step, epoch=True):
        """
        Validates the current model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        global_step: Epoch or step number.
        epoch: bool
            If True, this validation is at the end of an epoch.

        Return
        ------
        mean_valid_loss: float
        """
        self.model.eval()
        valid_loss = 0.
        valid_acc = 0.

        # validation loop
        for batch_idx, (image, target) in enumerate(self.valid_loader):
            with torch.inference_mode():
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)
                if self.in_codist:
                    loss = self.loss_fn(image, output, target)
                else:
                    loss = self.loss_fn(output, target)
                valid_loss += loss.item()
                valid_acc += accuracy(output, target).item()
                mean_valid_loss = valid_loss / (batch_idx + 1)
                mean_valid_acc = valid_acc / (batch_idx + 1)

                if epoch and self.can_print and (global_step + 1) % self.n_print == 0:
                    end = '' if batch_idx < len(self.valid_loader) - 1 else '\n\n'
                    print(
                        f'\rValidation accuracy: {mean_valid_acc:.6f} -- '
                        f'Avg. loss ({self.loss_fn.__name__}): {mean_valid_loss:.6f} -- '
                        f'Batch: {batch_idx + 1}/{len(self.valid_loader)} '
                        f'({100.*(batch_idx + 1) / len(self.valid_loader):.0f}%)',
                        end=end
                    )

        # plot or log feature maps and filters
        if epoch and self.n_plot > 0 and (global_step + 1) % self.n_plot == 0:
            _ = self.plot_or_log_activations(self.valid_loader, n_imgs=2, save_idxs=True, epoch=global_step,
                                             stage='validation', log=self.log)
            _ = self.plot_or_log_filters(save_idxs=True, epoch=global_step, log=self.log)

        if self.log:
            step_type = 'Epoch' if epoch else 'Step'
            self.writer.add_scalar(f'{step_type} loss/validation', mean_valid_loss, global_step)
            self.writer.add_scalar(f'{step_type} accuracy/validation', mean_valid_acc, global_step)

        return mean_valid_loss, mean_valid_acc


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
