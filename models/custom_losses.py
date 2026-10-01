# File: models/custom_losses.py

import torch
import torch.nn as nn

class AsymmetricLinexLoss(nn.Module):
    """
    Custom Linear-Exponential (Linex) Loss for Risk-Adjusted Trading.
    Penalizes over-predictions exponentially and under-predictions linearly.
    """
    def __init__(self, alpha: float = 1.0):
        super(AsymmetricLinexLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_pred - y_true
        # Linex formula: exp(a * e) - a*e - 1
        loss = torch.exp(self.alpha * error) - (self.alpha * error) - 1.0
        return torch.mean(loss)