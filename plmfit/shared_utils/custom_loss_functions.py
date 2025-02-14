import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, targets):
        EPS = 1e-12
        loss = -2 * (self.weights[0] * (targets * torch.log(pred + EPS)) + self.weights[1] * (1 - targets) * torch.log(1 - pred +EPS))
        return loss.mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Custom BCE loss that applies a sigmoid internally and ignores targets == -100.
    Use this if your model outputs raw logits (no sigmoid applied).
    """

    def __init__(self, reduction="mean", ignore_index=-100, pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        mask = targets != self.ignore_index

        filtered_logits = logits[mask]
        filtered_targets = targets[mask]

        # binary_cross_entropy_with_logits combines sigmoid and BCE in a numerically stable way
        loss = F.binary_cross_entropy_with_logits(
            filtered_logits, filtered_targets, reduction=self.reduction, pos_weight=self.pos_weight
        )
        return loss
