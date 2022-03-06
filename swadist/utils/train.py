"""Training utilities
Adapted from https://github.com/Yu-Group/adaptive-wavelets/blob/master/awave/utils/train.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from .losses import accuracy
from .viz import show_imgs

class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: optional, torch.model

    optimizer: torch.optim.Optimizer

    device: torch.device, optional
        Device on which to run the code.

    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 device='cpu',
                 n_print=1,
                 n_plot=5):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.n_print = n_print
        self.n_plot = n_plot

    def __call__(self, train_loader, valid_loader=None, epochs=10, swa_epochs=0):
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
        """
        print("Starting training Loop...")
        self.train_losses = np.empty(epochs + swa_epochs)
        self.valid_losses = np.empty(epochs + swa_epochs)
        for epoch in range(epochs + swa_epochs):
            mean_epoch_loss = self._train_epoch(train_loader, epoch)
            self.train_losses[epoch] = mean_epoch_loss
            if valid_loader:
                mean_epoch_valid_loss = self._validate_epoch(valid_loader, epoch)
                self.valid_losses[epoch] = mean_epoch_valid_loss
            elif epoch % self.n_print == 0:
                print('\n\n')
            if self.scheduler:
                self.scheduler.step()


    def _train_epoch(self, data_loader, epoch):
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
        n_images = 0
        for batch_idx, (image, target) in enumerate(data_loader):
            iter_loss = self._train_iteration(image, target)
            epoch_loss += iter_loss
            n_images += image.shape[0]
            if epoch % self.n_print == 0:
                end = '' if batch_idx < len(data_loader) - 1 else '\n'
                print(
                    f'\rTrain epoch: {epoch + 1} -- '
                    f'Avg. epoch loss ({self.loss_fn.__name__}): {epoch_loss / n_images:.6f} -- '
                    f'Batch: {batch_idx + 1}/{len(data_loader)} '
                    f'({100. * (batch_idx + 1) / len(data_loader):.0f}%) -- '
                    f'Total steps: {epoch*len(data_loader) + batch_idx + 1}',
                    end=end
                )
        if epoch % self.n_plot == 0:
            self.plot_feature_maps(data_loader, epoch)


        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss


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
        image, target = image.to(self.device), target.to(self.device)

        # clear gradients
        self.model.zero_grad()

        # calculate the loss & gradients
        output = self.model(image)
        loss = self.loss_fn(output, target)
        loss.backward()

        # update the optimizer step
        self.optimizer.step()

        return loss.item()


    def _validate_epoch(self, data_loader, epoch):
        """
        Validates the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        self.model.eval()
        epoch_loss = 0.
        epoch_acc = 0.
        for batch_idx, (image, target) in enumerate(data_loader):
            with torch.inference_mode():
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)
                loss = self.loss_fn(output, target)
                epoch_loss += loss.item()
                epoch_acc += accuracy(output, target)
                if epoch % self.n_print == 0:
                    end = '' if batch_idx < len(data_loader) - 1 else '\n\n'
                    print(
                        f'\r{self.loss_fn.__name__}: {epoch_loss / (batch_idx + 1):.6f} -- '
                        f'Accuracy: {epoch_acc / (batch_idx + 1):.6f} -- '
                        f'Batch: {batch_idx + 1}/{len(data_loader)} '
                        f'({100.*(batch_idx + 1) / len(data_loader):.0f}%)',
                        end=end
                    )
        if epoch % self.n_plot == 0:
            self.plot_feature_maps(data_loader, epoch, 'validation')
            if hasattr(self, 'filter_idx_dict'):
                imgs, _ = self.model.get_conv_imgs(self.filter_idx_dict)
            else:
                imgs, self.filter_idx_dict = self.model.get_conv_imgs()
            show_imgs(imgs, f'filters after epoch {epoch + 1}', [k for k in self.filter_idx_dict])

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss


    def plot_feature_maps(self, data_loader, epoch, stage='train'):
        self.model.eval()
        batch, _ = next(iter(data_loader))
        in_idx = np.random.choice(len(batch), replace=False)
        imgs = []
        imgs.append(data_loader.dataset.inv_transform(batch[in_idx]))
        for i, conv in enumerate(self.model.convs):
            out = self.model.partial_forward(batch, conv)
            conv_idx = np.random.choice(out.shape[1], replace=False)
            imgs.append(out[in_idx, conv_idx, :, :][None, :, :])
        titles = ['original']
        titles.extend(self.model.convs)
        show_imgs(imgs, f'{stage}, epoch {epoch + 1}', titles=titles)
