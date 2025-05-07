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

class MaskedBCELoss(nn.Module):
    """
    Custom BCE loss that assumes the inputs are probabilities (i.e. no sigmoid is applied)
    and ignores targets equal to ignore_index.
    """
    def __init__(self, reduction="mean", ignore_index=-100, pos_weight=None):
        super(MaskedBCELoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
            # Register the tensor so it moves with the model.
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, probs, targets, sample_weight=None):
        """
        probs:   Tensor of shape [batch_size, num_labels] with probabilities (in [0,1]).
        targets: Tensor of shape [batch_size, num_labels], where positions with the value 
                 `ignore_index` will be ignored.
        """
        # Create a mask for valid targets.
        mask = targets != self.ignore_index
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        # Clamp ignored targets to 0.0 so that BCE computation remains valid.
        clamped_targets = targets.clone()
        clamped_targets[~mask] = 0.0

        # To avoid log(0) issues, clamp probabilities to a small epsilon.
        eps = 1e-7
        probs = probs.clamp(min=eps, max=1 - eps)

        # Compute the element-wise BCE loss.
        # If pos_weight is provided, weight the positive examples accordingly.
        if self.pos_weight is not None:
            element_loss = - (clamped_targets * self.pos_weight * torch.log(probs) +
                              (1 - clamped_targets) * torch.log(1 - probs))
        else:
            element_loss = - (clamped_targets * torch.log(probs) +
                              (1 - clamped_targets) * torch.log(1 - probs))

        # Apply the mask to zero out loss for ignored targets.
        masked_loss = element_loss * mask.float()
        
        # Apply sample weights, if provided
        if sample_weight is not None:
            if sample_weight.ndim == 1:
                sample_weight = sample_weight.unsqueeze(1)
            masked_loss = masked_loss * sample_weight

        # Reduce the loss over the valid elements.
        if self.reduction == "mean":
            loss = masked_loss.sum() / mask.sum().float()
        elif self.reduction == "sum":
            loss = masked_loss.sum()
        else:  # "none": return the full loss matrix.
            loss = masked_loss

        return loss

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Custom BCE loss that applies a sigmoid internally and ignores targets == -100.
    Use this if your model outputs raw logits (no sigmoid applied).
    """

    def __init__(self, reduction="mean", ignore_index=-100, pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
            # Use register_buffer so this tensor is moved along with the model
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, logits, targets, sample_weight=None):
        """
        logits:   shape [batch_size, num_labels]
        targets:  shape [batch_size, num_labels], with ignore_index in some places
        """
        mask = targets != self.ignore_index

        # We must clamp "ignored" targets to something valid for BCE (e.g., 0)
        # so that the BCE won't produce NaNs.
        clamped_targets = targets.clone()
        clamped_targets[~mask] = 0.0  # or 1.0, doesn't matter, because they'll be masked out

        # Apply sample weights, if provided
        element_loss = F.binary_cross_entropy_with_logits(
            logits, clamped_targets,
            pos_weight=self.pos_weight,
            weight=sample_weight,
            reduction="none"  # shape [batch_size, num_labels]
        )

        # Zero out ignored labels
        masked_loss = element_loss * mask 


        # 3) Reduce over the valid elements only
        if self.reduction == "mean":
            loss = masked_loss.sum() / mask.sum()
        elif self.reduction == "sum":
            loss = masked_loss.sum()
        else:
            # e.g., "none" – return the full matrix
            loss = masked_loss

        return loss

class MaskedFocalWithLogitsLoss(nn.Module):
    """
    Focal loss extended for use with missing labels. Ignores targets == ignore_index.
    Reference (original focal loss): https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2, reduction="mean", ignore_index=-100):
        """
        Args:
            alpha (float): Weighting factor for positives in [0, 1]. 
                           If alpha >= 0, alpha is applied to positive examples, 
                           (1 - alpha) to negatives.
            gamma (float): Exponent of the modulating factor (1 - p_t)**gamma to reduce 
                           the relative loss for well-classified examples.
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): Specifies a target value that is ignored and does not 
                                contribute to the gradient.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, ...) Tensor of raw predictions (before sigmoid).
            targets: (N, ...) Tensor of same shape as logits, 
                     with 0 or 1 for valid labels, and ignore_index for missing.

        Returns:
            Loss tensor of shape (), or the same shape as inputs if reduction='none'.
        """
        # Build the mask for valid labels (those != ignore_index)
        mask = (targets != self.ignore_index)

        # Clone targets and clamp ignored entries to some valid label (0 or 1).
        # We'll use 0 here, but since they are masked out, it won't matter.
        clamped_targets = targets.clone()
        clamped_targets[~mask] = 0.0

        # Compute the standard binary cross-entropy per element, no reduction
        ce_loss = F.binary_cross_entropy_with_logits(logits, clamped_targets, reduction="none")

        # Compute p = sigmoid(logits)
        p = torch.sigmoid(logits)

        # p_t = p * t + (1 - p) * (1 - t)
        #    This is the probability assigned to the *true* class per element
        p_t = p * clamped_targets + (1 - p) * (1 - clamped_targets)

        # Apply the focal term: (1 - p_t)**gamma
        focal_term = (1.0 - p_t) ** self.gamma
        loss = ce_loss * focal_term

        # If alpha >= 0, compute alpha_t = alpha for positives, (1-alpha) for negatives
        if self.alpha >= 0:
            alpha_t = self.alpha * clamped_targets + (1.0 - self.alpha) * (1.0 - clamped_targets)
            loss = alpha_t * loss

        # Mask out the ignored entries
        loss = loss * mask

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            # Average over the valid (non-ignored) entries
            valid_count = mask.sum()
            if valid_count == 0:
                # If there are no valid entries, return zero (or you could raise a warning).
                return loss.sum() * 0.0
            else:
                return loss.sum() / valid_count
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid 'reduction' mode: {self.reduction}. "
                "Supported: 'none', 'mean', 'sum'."
            )
