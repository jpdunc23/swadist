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
           'CodistillationLoss']


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
    async_op: bool, optional
        If True, update replica parameters asynchronously and continue with parameters from the
        last time they were updated.
    transform: str, optional
        Name of a method to transform the mean replica output before calling `loss_fn`.
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
        self.swa_model = None
        self._handle = None
        self._flat_params = None
        self._debug = debug

        for i in range(self.world_size):
            if i == self.rank:
                self.replicas.append(self.model)
            else:
                self.replicas.append(deepcopy(self.model))

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

            update_replicas = (self.update_swa_replicas
                               if self.swa_model is not None
                               else self.update_replicas)

            if self._handle is not None:
                warnings.warn(
                    f'Rank {self.rank} at codist step {self.step}: It\'s taking longer than '
                    f'sync_freq={self.sync_freq} steps to gather replica parameters. Calling wait() '
                    'on async work handle.'
                )
                update_replicas(force=True)
            else:
                update_replicas()

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
                    mean_output += model(x) / (self.world_size - 1)

        return mean_output


    def _all_gather_params(self):

        if self._debug:
            print(f'\nRank {self.rank} gathering replica params at codist step {self.step}')

        # construct the Tensor list to all_gather into
        params = self.replicas[self.rank].parameters()

        # flat_params[i] is a single Tensor with all params from rank i
        self._handle, self._flat_params_list = all_gather(*params,
                                                          rank=self.rank,
                                                          world_size=self.world_size,
                                                          async_op=self.async_op)


    def update_replicas(self, force=False):
        """Starts comm to update replicas, if need be, and updates replicas if comm has
        finished and returns True, or returns False if comm still in progress.

        """

        if self._handle is None:
            # begin comm
            self._all_gather_params()

        if force or not self.async_op:
            # wait for comm to finish
            self._handle.wait()

        # update params if comm has finished
        if self._handle.is_completed():

            if self._debug:
                print(f'\nRank {self.rank} updating replica params at codist step {self.step}')

            # update the replicas' parameters
            for i, flat_params in enumerate(self._flat_params_list):
                if i != self.rank:
                    vector_to_parameters(flat_params, self.replicas[i].parameters())

            self._handle = None
            self._flat_params_list = None

            return True

        return False


    def update_swa_replicas(self, swa_model: torch.optim.swa_utils.AveragedModel=None,
                            force=False):
        """Add or update (optionally) `self.swa_model`, which is used to populate replicas once added. update
        replicas, and all_gather model params synchronously.

        """

        assert swa_model is not None or self.swa_model is not None, \
            'When first calling update_swa_replicas, "swa_model" arg cannot be None.'

        if self._debug:
            print(f'\nRank {self.rank} updating SWA replica params at codist step {self.step}')

        # if no update to self.swa_model, then update the replica using self.model
        update_avg = swa_model is None

        # always force update replicas if self.swa_model was updated
        force = force or not update_avg

        if update_avg:

            # get a fresh copy of the AveragedModel
            self.replicas[self.rank] = deepcopy(self.swa_model)

            # add current model to the average
            self.replicas[self.rank].update_parameters(self.model)

            # TODO: update_bn?

        elif swa_model is not self.swa_model:
                # need to (re)create all replicas

            self.swa_model = swa_model
            self.replicas = []

            for i in range(self.world_size):
                replica = deepcopy(self.swa_model)
                self.replicas.append(replica)

        else:

            # just need to make clean copy of this rank's replica
            self.replicas[self.rank] = deepcopy(self.swa_model)

        self.update_replicas(force=force)
