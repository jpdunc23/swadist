"""Loss functions and non-grad evaluation metrics.
"""

import warnings

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import vector_to_parameters

from .distributed import all_gather


__all__ = ['accuracy',
           'CodistillationLoss',
           'SWADistLoss']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Adapted from https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k / batch_size)
        if len(res) == 1:
            return res[0]
        return res


_transform_fns = {
    'argmax': lambda x: x.argmax(dim=1),
    'softmax' : lambda x: x.softmax(dim=1),
}


class CodistillationLoss(object):
    """Manages model replicas and computes codistillation loss.

    Parameters
    __________
    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]
        Differentiable loss function to use when comparing output to average outputs from
        replicas.
    model: torch.nn.Module
        Model to replicate.
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    world_size: int
        The number of distributed workers. If greater than 1, then .
    sync_freq: int, optional
        Frequency of replica updates across ranks, in number of training steps. If `sync_replicas`
        is 0, replicas are sync'ed only once (at initialization), but they can later be updated
        manually using `update_replicas()`, with `force=True` to block if `async_op` is True.
    transform: str, optional
        Name of a method to transform the mean replica output before calling `loss_fn`.
    async_op: bool, optional
        If True, update replica parameters asynchronously and continue with parameters from the
        last time they were updated.
    debug: bool, optional
        If True, print step when gathering parameters begins and when the parameters are ready.

    """
    def __init__(self,
                 loss_fn,
                 model,
                 rank,
                 world_size,
                 sync_freq=50,
                 transform=None,
                 async_op=True,
                 debug=False):

        self.__name__ = 'CodistillationLoss'
        self.loss_fn = loss_fn
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.sync_freq = sync_freq
        self.transform = transform
        self.async_op = async_op
        self.step = 0
        self.replicas = []
        self.debug = debug
        self._handle = None
        self._flat_params = None

        for i in range(self.world_size):
            if i == self.rank:
                self.replicas.append(self.model)
            else:
                self.replicas.append(deepcopy(self.model))
                for param in self.replicas[i].parameters():
                    param.detach()

        # block on cpu until all ranks are ready
        torch.cuda.synchronize()

        # force update replicas
        self.update_replicas(force=True)


    def __call__(self, x, output):
        """Calculate codistillation component of loss, i.e. the original loss function
        of model output and average model outputs from other replicas in the collective.

        Parameters
        __________
        x: torch.Tensor
            A batch of input observations.
        output: torch.Tensor
            A batch of model outputs from this rank's model.

        """

        # update the step counter
        self.step += 1

        if min(self.sync_freq, self.step - 1) > 0 and (self.step - 1) % self.sync_freq == 0:

            if self._handle is not None:
                warnings.warn(
                    f'Rank {self.rank} at codist step {self.step}: It\'s taking longer than '
                    f'sync_freq={self.sync_freq} steps to gather replica parameters. '
                    'Calling wait() on the async work handle.'
                )
                self.update_replicas(force=True)
            else:
                self.update_replicas()

        mean_output = self.replica_mean_output(x, output)

        if self.transform is not None:
            mean_output = _transform_fns[self.transform](mean_output)

        return self.loss_fn(output, mean_output)


    def replica_mean_output(self, x, output):
        """Get mean predictions from saved model replicas.
        """

        if self._handle is not None:
            # check on status of update
            self.update_replicas()

        with torch.no_grad():

            # average over other ranks
            mean_output = torch.empty_like(output)

            for i, model in enumerate(self.replicas):

                if i != self.rank:
                    model.eval()
                    mean_output += model(x) / (self.world_size - 1)

        return mean_output


    def _all_gather_params(self, force=None):

        if force == None:
            force = self.async_op

        if self.debug:
            print(f'\nRank {self.rank} gathering replica params at codist step {self.step}')

        # construct the Tensor list to all_gather into
        params = self.replicas[self.rank].parameters()

        # flat_params[i] is a single Tensor with all params from rank i
        self._handle, self._flat_params_list = all_gather(*params,
                                                          rank=self.rank,
                                                          world_size=self.world_size,
                                                          async_op=force)


    def update_replicas(self, force=False):
        """Starts comm to update replicas, if need be, and updates replicas if comm has
        finished.

        """

        if self._handle is None:
            # begin comm
            self._all_gather_params(force=force)

        # if force or not self.async_op:
        #     # wait for comm to finish
        #     self._handle.wait()

        # update params if comm has finished
        if self._handle.is_completed():

            if self.debug:
                print(f'\nRank {self.rank} updating replica params at codist step {self.step}')

            # update the replicas' parameters
            for i, flat_params in enumerate(self._flat_params_list):
                if i != self.rank:
                    vector_to_parameters(flat_params, self.replicas[i].parameters())

            self._handle = None
            self._flat_params_list = None


class SWADistLoss(object):
    """Manages `torch.optim.swa_utils.AveragedModel` replicas and computes codist loss.

    Parameters
    __________
    loss_fn: Union[Callable, torch.nn.modules.loss._Loss]
        Differentiable loss function to use when comparing output to average outputs from
        replicas.
    model: torch.nn.Module
        Model which gets updated by call to `backward()` on returned loss.
    swa_model: torch.optim.swa_utils.AveragedModel
        SWA models to replicate.
    rank: int
        The index of the current worker, from `0` to `world_size - 1`.
    world_size: int
        The number of distributed workers. If greater than 1, then .
    sync_freq: int, optional
        Frequency of replica updates across ranks, in number of training steps. If `sync_replicas`
        is 0, replicas are sync'ed only once (at initialization), but they can later be updated
        manually using `update_replicas()`, with `force=True` to block if `async_op` is True.
    transform: str, optional
        Name of a method to transform the mean replica output before calling `loss_fn`.
    async_op: bool, optional
        If True, update replica parameters asynchronously and continue with parameters from the
        last time they were updated.
    debug: bool, optional
        If True, print step when gathering parameters begins and when the parameters are ready.

    """
    def __init__(self,
                 loss_fn,
                 model,
                 swa_model,
                 rank,
                 world_size,
                 sync_freq=50,
                 transform=None,
                 async_op=True,
                 debug=False):

        self.__name__ = 'SWADistLoss'
        self.loss_fn = loss_fn
        self.model = model
        self.swa_model = swa_model
        self.rank = rank
        self.world_size = world_size
        self.sync_freq = sync_freq
        self.async_op = async_op
        self.step = 0
        self.debug = debug
        self._codist_loss = CodistillationLoss(loss_fn,
                                               deepcopy(swa_model),
                                               rank,
                                               world_size,
                                               sync_freq,
                                               transform,
                                               async_op,
                                               debug)


    def __call__(self, x, output):
        """Calculate SWADist component of loss, i.e. the original loss function applied
        to the model output and the mean of the `torch.optim.swa_utils.AveragedModel`
        outputs from the other ranks in the collective.

        Parameters
        __________
        x: torch.Tensor
            A batch of input observations.
        output: torch.Tensor
            A batch of model outputs from this rank's model.

        """
        self.step += 1

        if min(self.sync_freq, self.step - 1) > 0 and (self.step - 1) % self.sync_freq == 0:
            # add self.model params to this rank's AveragedModel replica before syncing
            self.update_parameters()

        loss = self._codist_loss(x, output)

        return loss


    def update_replicas(self):
        """Update swa_model replicas with copies of `self.swa_model`. Should be called
        after calling `self.swa_model.update_parameters()`.

        """

        if self.debug:
            print(f'\nRank {self.rank} updating replicas at swadist step {self._codist_loss.step}')

        self._codist_loss.replicas = []
        for i in range(self.world_size):
            self._codist_loss.replicas.append(deepcopy(self.swa_model))

        # synchronize so all ranks have the updated versions of `self.swa_model`.
        self._codist_loss.update_replicas(force=True)


    def update_parameters(self):
        """Updates this rank's `AveragedModel` replica with the current parameters of `self.model`.

        """

        if self.debug:
            print(f'\nRank {self.rank} updating SWA replica params '
                  f'at swadist step {self._codist_loss.step}')

        # get a fresh copy of this ranks AveragedModel
        self._codist_loss.replicas[self.rank] = deepcopy(self.swa_model)

        # add current model to the average
        self._codist_loss.replicas[self.rank].update_parameters(self.model)

        # TODO: update_bn?
