import torch
import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="mlloss")
class MLLoss(nn.Module):
    def __init__(self):
        super(MLLoss, self).__init__()

    def forward(self, input, target, eps=1e-6):
        # 0 - real; 1 - fake.
        loss = torch.tensor(0., device=target.device)
        batch_size = target.shape[0]
        mat_1 = torch.hstack([target.unsqueeze(-1)] * batch_size)
        mat_2 = torch.vstack([target] * batch_size)
        diff_mat = torch.logical_xor(mat_1, mat_2).float()
        or_mat = torch.logical_or(mat_1, mat_2)
        eye = torch.eye(batch_size, device=target.device)
        or_mat = torch.logical_or(or_mat, eye).float()
        sim_mat = 1. - or_mat
        for _ in input:
            diff = torch.sum(_ * diff_mat, dim=[0, 1]) / (torch.sum(diff_mat, dim=[0, 1]) + eps)
            sim = torch.sum(_ * sim_mat, dim=[0, 1]) / (torch.sum(sim_mat, dim=[0, 1]) + eps)
            partial_loss = 1. - sim + diff
            loss += max(partial_loss, torch.zeros_like(partial_loss))
        return loss