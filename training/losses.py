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
    
    IMPORTANT: If `log_vars` (learned uncertainty) is used (requires_grad=True), 
    the manual `weights` dictionary IS IGNORED to prevent double-weighting ambiguity.
    Uncertainty weighting automatically handles scaling differences (e.g. MSE vs BCE).
    """

    def __init__(
        self,
        rise_fall_weight: float = 1.0,
        touch_weight: float = 0.5,
        range_weight: float = 0.5,
        reconstruction_weight: float = 50.0,
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

        # Uncertainty weighting parameters (Kendall et al. 2018)
        # log(sigma^2) for each task: rise_fall, touch, range, reconstruction
        self.log_vars = nn.Parameter(torch.zeros(4))

    def compute_weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted total loss using learned uncertainty weights.
        
        Formula: L = sum( exp(-s_i) * L_i + 0.5 * s_i )
        where s_i = log(sigma_i^2)
        """
        task_keys = ["rise_fall", "touch", "range", "reconstruction"]
        # Use device of first loss tensor found, or cpu
        first_loss = next(iter(losses.values())) if losses else torch.tensor(0.0)
        total = torch.tensor(0.0, device=first_loss.device)
        
        for i, key in enumerate(task_keys):
            if key in losses:
                # Weighted loss term
                precision = torch.exp(-self.log_vars[i])
                total += precision * losses[key] + 0.5 * self.log_vars[i]
                
        return total

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
        # 1. Compute RAW (unweighted) losses for all active heads
        
        if "rise_fall_logit" in logits and "rise_fall" in targets:
            losses["rise_fall"] = self.bce(logits["rise_fall_logit"].squeeze(), targets["rise_fall"].float())

        if "touch_logit" in logits and "touch" in targets:
            losses["touch"] = self.bce(logits["touch_logit"].squeeze(), targets["touch"].float())

        if "range_logit" in logits and "range" in targets:
            losses["range"] = self.bce(logits["range_logit"].squeeze(), targets["range"].float())

        # Autoencoder reconstruction loss
        recon_out = vol_reconstruction if vol_reconstruction is not None else logits.get("vol_reconstruction")
        if vol_input is not None and recon_out is not None:
            losses["reconstruction"] = self.mse(recon_out, vol_input)

        # 2. Compute Total Loss based on Weighting Strategy
        
        # Initialize total as tensor on correct device
        first_loss = next(iter(losses.values())) if losses else torch.tensor(0.0)
        losses["total"] = torch.tensor(0.0, device=first_loss.device)

        if hasattr(self, "log_vars") and self.log_vars.requires_grad:
            # STRATEGY A: Learned Uncertainty Weighting
            # Ignores self.weights
            losses["total"] = self.compute_weighted_loss(losses)
        else:
            # STRATEGY B: Manual Weighting
            # Applies self.weights
            for key in ["rise_fall", "touch", "range", "reconstruction"]:
                if key in losses:
                    losses["total"] += losses[key] * self.weights.get(key, 1.0)
                    # Update the individual entry in losses dict to reflect the WEIGHTED value?
                    # Generally useful for logging to see actual contribution.
                    losses[key] = losses[key] * self.weights.get(key, 1.0)

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
