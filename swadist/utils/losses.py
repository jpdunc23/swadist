"""Loss functions and evaluation utilities
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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


class CodistillationLoss():
    def __init__(self,
                 loss_fn,
                 model,
                 device,
                 rank,
                 world_size,
                 sync_freq=50):
        self.__name__ = 'CodistillationLoss'
        self.loss_fn = loss_fn
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.replicas = []
        self.sync_freq = sync_freq
        self.step = 0

        for i in range(world_size):
            if i == self.rank:
                self.replicas.append(model)
            else:
                self.replicas.append(deepcopy(model))


    def __call__(self, x, output):

        if self.step % self.sync_freq == 0:
            self.update_replicas()

            # update the step counter
        self.step += 1

        # get avg. predictions from the other models
        # with torch.no_grad():
        with torch.no_grad():
            # average over other ranks
            avg_output = torch.zeros(output.shape).to(self.device)
            for i, model in enumerate(self.replicas):
                if i != self.rank:
                    avg_output += model(x) / (self.world_size - 1)
            avg_output = avg_output.softmax(dim=1)

        return self.loss_fn(output, avg_output)


    def update_replicas(self):

        # construct the Tensor list to all_gather into
        state_dicts = [m.state_dict() for m in self.replicas]
        handles = []
        for key, param in state_dicts[0].items():
            # this is where the rest of group will send their params
            recv_list = [torch.zeros(param.shape).to(self.device)
                         for _ in range(self.world_size)]

            # send my param to the rest of the group
            dist.all_gather(recv_list, param)

            # update the replica state dicts
            for i, new_param in enumerate(recv_list):
                if i != self.rank:
                    state_dicts[i][key] = new_param.detach()

        # load the new state dicts
        for i, model in enumerate(self.replicas):
            if i != self.rank:
                model.load_state_dict(state_dicts[i])


