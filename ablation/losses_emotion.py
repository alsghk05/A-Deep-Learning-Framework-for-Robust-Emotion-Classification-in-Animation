# losses_emotion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOSS_NAME, LABEL_SMOOTHING, FOCAL_GAMMA


class FocalLoss(nn.Module):
    """
    multi-class focal loss
    """
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C), target: (B,)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # one-hot target
        target_one_hot = F.one_hot(target, num_classes=logits.size(1)).float()
        pt = (probs * target_one_hot).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.nll_loss(log_probs, target, reduction="none")
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss():
    """
    config의 LOSS_NAME에 따라 loss 반환
    """
    if LOSS_NAME == "ce":
        return nn.CrossEntropyLoss()

    elif LOSS_NAME == "ce_ls":
        # label smoothing CrossEntropy
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    elif LOSS_NAME == "focal":
        return FocalLoss(gamma=FOCAL_GAMMA)

    else:
        raise ValueError(f"지원하지 않는 LOSS_NAME: {LOSS_NAME}")
