"""
Custom loss functions for multi-expert trading model.
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Combined loss for all contract heads + autoencoder reconstruction.

    Components:
    - BCE loss for each contract head (rise_fall, touch, range)
    - MSE reconstruction loss for volatility autoencoder

    Weights can be adjusted to prioritize certain tasks.
    """

    def __init__(
        self,
        rise_fall_weight: float = 1.0,
        touch_weight: float = 0.5,
        range_weight: float = 0.5,
        reconstruction_weight: float = 0.1,
    ):
        super().__init__()
        self.weights = {
            "rise_fall": rise_fall_weight,
            "touch": touch_weight,
            "range": range_weight,
            "reconstruction": reconstruction_weight,
        }

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        vol_input: torch.Tensor | None = None,
        vol_reconstruction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            logits: Dict of model outputs {head_name: logit_tensor}
            targets: Dict of target labels {head_name: target_tensor}
            vol_input: Original volatility metrics (for reconstruction loss)
            vol_reconstruction: Autoencoder reconstruction output

        Returns:
            Dict with 'total', 'rise_fall', 'touch', 'range', 'reconstruction'
        """
        losses = {}

        # Contract head losses
        if "rise_fall_logit" in logits and "rise_fall" in targets:
            losses["rise_fall"] = (
                self.bce(logits["rise_fall_logit"].squeeze(), targets["rise_fall"].float())
                * self.weights["rise_fall"]
            )

        if "touch_logit" in logits and "touch" in targets:
            losses["touch"] = (
                self.bce(logits["touch_logit"].squeeze(), targets["touch"].float())
                * self.weights["touch"]
            )

        if "range_logit" in logits and "range" in targets:
            losses["range"] = (
                self.bce(logits["range_logit"].squeeze(), targets["range"].float())
                * self.weights["range"]
            )

        # Autoencoder reconstruction loss
        if vol_input is not None and vol_reconstruction is not None:
            losses["reconstruction"] = (
                self.mse(vol_reconstruction, vol_input) * self.weights["reconstruction"]
            )

        # Total loss
        losses["total"] = sum(losses.values())

        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in trading signals.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"

    Reduces loss for well-classified examples, focusing on hard cases.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Binary targets (0 or 1)
        """
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()
