import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, targets):
        EPS = 1e-12
        loss = -2 * (self.weights[0] * (targets * torch.log(pred + EPS)) + self.weights[1] * (1 - targets) * torch.log(1 - pred +EPS))
        return loss.mean()

"""
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, pred, targets):
        loss = self.loss_fn(pred, targets, reduction = 'none')
        loss_mask = torch.ones_like(targets)
        loss_mask[targets == -1] = 0
        loss_weighted = torch.mul(loss , loss_mask).sum(dim = 1)

        return loss_weighted.mean()