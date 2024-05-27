import torch
import torch.nn as nn

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