"""Training utilities
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from .losses import accuracy
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
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 device='cpu',
                 name='trainer',
                 n_print=1,
                 n_plot=0,
                 log=False,
                 log_dir='./runs'):
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
        if self.log:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(f'{log_dir}/{self.name}_{timestamp}')


    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)


    def train(self, train_loader, valid_loader=None, epochs=10, swa_epochs=0,
              codistill=False, stopping_acc=np.inf, validations_per_epoch=None):
        """
        Trains the model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
        valid_loader: torch.utils.data.DataLoader, optional
        epochs: int, optional
            Number of epochs to train the model for.
        swa_epochs: int, optional
            Number of additional epochs for stochastic weight averaging.
        codistill: bool, optional
            If True, use codistillation.
        stopping_acc: float, optional
            Validation accuracy at which to stop training.
        """
        print("Starting {epochs}-epoch training loop...")

        self.train_epochs = epochs + swa_epochs
        self.train_losses = np.empty(epochs + swa_epochs)
        self.train_accs = np.empty(epochs + swa_epochs)
        self.valid_losses = np.empty(epochs + swa_epochs)
        self.valid_accs = np.empty(epochs + swa_epochs)

        for epoch in range(epochs + swa_epochs):
            if epoch == epochs:
                # save the model at end of first phase and create the SWA model
                self.base_model = self.model
                self.model = torch.optim.swa_utils.AveragedModel(self.base_model)

            # epoch loop
            mean_epoch_loss, mean_epoch_acc = self._train_epoch(train_loader,
                                                                epoch,
                                                                valid_loader,
                                                                validations_per_epoch)
            self.train_losses[epoch] = mean_epoch_loss
            self.train_accs[epoch] = mean_epoch_acc

            if epoch > epochs:
                # update running average of parameters
                self.model.update_parameters(self.model)

            if valid_loader:

                # update bn statistics before validation
                if swa_epochs > 0:
                    torch.optim.swa_utils.update_bn(train_loader, self.model)

                # validate the epoch
                mean_valid_loss, mean_valid_acc = self._validate(valid_loader, global_step=epoch)
                self.valid_losses[epoch] = mean_valid_loss
                self.valid_accs[epoch] = mean_valid_acc

                if self.log:
                    global_step = len(train_loader)*(epoch + 1)
                    self.writer.add_scalar('Step loss/validation', mean_valid_loss, global_step)
                    self.writer.add_scalar('Step accuracy/validation', mean_valid_acc, global_step)

                # stop early?
                if mean_valid_acc >= stopping_acc:
                    print(f'Validation accuracy target reached. Stopping early after {epoch + 1} epochs'
                          f' ({global_step} steps)')
                    self.train_epochs = epoch + 1
                    break

            elif self.n_print > 0 and (epoch + 1) % self.n_print == 0:
                print('\n')

            if self.scheduler:
                self.scheduler.step()

        # update bn statistics at the end of training if needed
        if swa_epochs > 0 and not valid_loader:
            torch.optim.swa_utils.update_bn(train_loader, self.model)

        if self.log:
            self.writer.close()


    def _train_epoch(self, data_loader, epoch, valid_loader=None, n_valid=None):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        self.model.train()
        epoch_loss = 0.
        epoch_acc = 0.
        valid_freq = None

        # calculate frequency of validation passes
        if valid_loader and n_valid:
            n_valid = min(len(data_loader), n_valid) - 1
            if n_valid > 0:
                valid_freq = int(np.floor(len(data_loader) / n_valid))

        # training loop
        for batch_idx, (image, target) in enumerate(data_loader):
            step = epoch*len(data_loader) + batch_idx + 1
            image, target = image.to(self.device), target.to(self.device)
            iter_loss, iter_acc = self._train_iteration(image, target)
            epoch_loss += iter_loss
            epoch_acc += iter_acc

            if self.log:
                self.writer.add_scalar('Step loss/training', iter_loss, step)
                self.writer.add_scalar('Step accuracy/training', iter_acc, step)

            if self.n_print > 0 and (epoch + 1) % self.n_print == 0:
                end = '' if batch_idx < len(data_loader) - 1 else '\n'
                print(
                    f'\rTrain epoch: {epoch + 1} -- '
                    f'Accuracy: {epoch_acc / (batch_idx + 1):.6f} -- '
                    f'Avg. loss ({self.loss_fn.__name__}): {epoch_loss / (batch_idx + 1):.6f} -- '
                    f'Batch: {batch_idx + 1}/{len(data_loader)} '
                    f'({100. * (batch_idx + 1) / len(data_loader):.0f}%) -- '
                    f'Total steps: {step}',
                    end=end
                )

            # validate every valid_freq steps
            if batch_idx < len(data_loader) - 1 and valid_freq and (batch_idx + 1) % valid_freq == 0:
                _, _ = self._validate(valid_loader, step, epoch=False)

        if self.n_plot > 0 and (epoch + 1) % self.n_plot == 0:
            _ = self.plot_or_log_activations(data_loader, n_imgs=2, save_idxs=True,
                                         epoch=epoch, log=self.log)

        mean_epoch_loss = epoch_loss / len(data_loader)
        mean_epoch_acc = epoch_acc / len(data_loader)

        if self.log:
            self.writer.add_scalar('Epoch loss/training', mean_epoch_loss, epoch)
            self.writer.add_scalar('Epoch accuracy/training', mean_epoch_acc, epoch)

        return mean_epoch_loss, mean_epoch_acc


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
        loss = self.loss_fn(output, target)
        loss.backward()

        # update the optimizer step
        self.optimizer.step()

        # calculate accuracy
        acc = accuracy(output, target)

        return loss.item(), acc.item()


    def _validate(self, data_loader, global_step, epoch=True):
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
        for batch_idx, (image, target) in enumerate(data_loader):
            with torch.inference_mode():
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)
                loss = self.loss_fn(output, target)
                valid_loss += loss.item()
                valid_acc += accuracy(output, target).item()
                mean_valid_loss = valid_loss / (batch_idx + 1)
                mean_valid_acc = valid_acc / (batch_idx + 1)

                if epoch and self.n_print > 0 and (global_step + 1) % self.n_print == 0:
                    end = '' if batch_idx < len(data_loader) - 1 else '\n\n'
                    print(
                        f'\rValidation accuracy: {mean_valid_acc:.6f} -- '
                        f'Avg. loss ({self.loss_fn.__name__}): {mean_valid_loss:.6f} -- '
                        f'Batch: {batch_idx + 1}/{len(data_loader)} '
                        f'({100.*(batch_idx + 1) / len(data_loader):.0f}%)',
                        end=end
                    )

        # plot or log feature maps and filters
        if epoch and self.n_plot > 0 and (global_step + 1) % self.n_plot == 0:
            _ = self.plot_or_log_activations(data_loader, n_imgs=2, save_idxs=True, epoch=global_step,
                                             stage='validation', log=self.log)
            _ = self.plot_or_log_filters(save_idxs=True, epoch=global_step, log=self.log)

        if self.log:
            step_type = 'Epoch' if epoch else 'Step'
            self.writer.add_scalar(f'{step_type} loss/validation', mean_valid_loss, global_step)
            self.writer.add_scalar(f'{step_type} accuracy/validation', mean_valid_acc, global_step)

        return mean_valid_loss, mean_valid_acc


    def plot_or_log_activations(self, data_loader, img_idx_dict=None, n_imgs=None, save_idxs=False,
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
                img_idxs = np.random.choice(len(data_loader.dataset), size=n_imgs, replace=False)
                img_idxs.sort()
                img_idx_dict['img_idxs'] = img_idxs
        else:
            img_idxs = img_idx_dict['img_idxs']

        for i, img_idx in enumerate(img_idxs):
            imgs = []
            image = data_loader.dataset[img_idx][0]
            original = data_loader.dataset.inv_transform(image)
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

            if log:
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

        if log:
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
