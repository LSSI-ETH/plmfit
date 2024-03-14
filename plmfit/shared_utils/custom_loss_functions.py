import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, targets):
        EPS = 1e-12
        loss = -2 * (self.weights[0] * (targets * torch.log(pred + EPS)) + self.weights[1] * (1 - targets) * torch.log(1 - pred +EPS))
        return loss.mean()