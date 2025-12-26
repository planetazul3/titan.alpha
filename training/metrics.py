"""
Trading-specific metrics for model evaluation.
"""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TradingMetrics:
    """
    Metrics tracker for trading model evaluation.

    Tracks:
    - Classification metrics (accuracy, precision, recall, F1)
    - Probability calibration (Brier score)
    - Trading-specific metrics (profit factor, win rate)
    """

    predictions: list[float] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)

    def reset(self):
        """Clear accumulated predictions."""
        self.predictions.clear()
        self.targets.clear()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Add batch predictions.

        Args:
            preds: Predicted probabilities (after sigmoid)
            targets: Ground truth labels (0 or 1)
        """
        self.predictions.extend(preds.detach().cpu().numpy().flatten().tolist())
        self.targets.extend(targets.detach().cpu().numpy().flatten().astype(int).tolist())

    def compute(self, threshold: float = 0.5) -> dict[str, float]:
        """
        Compute all metrics.

        Args:
            threshold: Classification threshold

        Returns:
            Dict of metric names to values
        """
        if len(self.predictions) == 0:
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        # Binary predictions
        binary_preds = (preds >= threshold).astype(int)

        # Basic counts
        tp = np.sum((binary_preds == 1) & (targets == 1))
        tn = np.sum((binary_preds == 0) & (targets == 0))
        fp = np.sum((binary_preds == 1) & (targets == 0))
        fn = np.sum((binary_preds == 0) & (targets == 1))

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Win rate (for trading)
        win_rate = recall  # Same as recall for binary win/loss

        # Brier score (probability calibration)
        brier = np.mean((preds - targets) ** 2)

        # ROC-AUC (simplified)
        auc = self._compute_auc(preds, targets)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "win_rate": win_rate,
            "brier_score": brier,
            "auc": auc,
            "total_samples": len(targets),
            "positive_rate": np.mean(targets),
        }

    def _compute_auc(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute ROC-AUC using trapezoidal rule.
        """
        # Handle edge cases
        if len(np.unique(targets)) < 2:
            return 0.5

        # Sort by prediction (descending)
        sorted_idx = np.argsort(-preds)
        sorted_targets = targets[sorted_idx]

        # Compute TPR and FPR at each threshold
        n_pos = np.sum(targets == 1)
        n_neg = np.sum(targets == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = np.cumsum(sorted_targets) / n_pos
        fpr = np.cumsum(1 - sorted_targets) / n_neg

        # Prepend (0, 0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Trapezoidal integration
        if hasattr(np, "trapezoid"):
            auc = np.trapezoid(tpr, fpr)
        else:
            auc = getattr(np, "trapz")(tpr, fpr)

        return float(auc)


def compute_profit_factor(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5, payout_ratio: float = 0.9
) -> float:
    """
    Compute profit factor for binary options.

    Profit Factor = Gross Profit / Gross Loss

    For binary options:
    - Win: +payout_ratio (e.g., +0.9 for 90% payout)
    - Loss: -1.0 (lose stake)

    Args:
        predictions: Model probabilities
        targets: Actual outcomes (1=win, 0=loss)
        threshold: Trading threshold
        payout_ratio: Payout on win (e.g., 0.9)

    Returns:
        Profit factor (>1 is profitable)
    """
    # Only trade when prediction meets threshold
    trades = predictions >= threshold

    if not np.any(trades):
        return 0.0

    wins = trades & (targets == 1)
    losses = trades & (targets == 0)

    gross_profit = np.sum(wins) * payout_ratio
    gross_loss = np.sum(losses) * 1.0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return cast(float, gross_profit / gross_loss)


def compute_expected_value(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5, payout_ratio: float = 0.9
) -> float:
    """
    Compute expected value per trade.

    EV = (win_rate * payout) - (loss_rate * 1.0)

    Args:
        predictions: Model probabilities
        targets: Actual outcomes
        threshold: Trading threshold
        payout_ratio: Payout on win

    Returns:
        Expected value (-1 to +payout_ratio)
    """
    trades = predictions >= threshold

    if not np.any(trades):
        return 0.0

    traded_targets = targets[trades]
    win_rate = np.mean(traded_targets)
    loss_rate = 1 - win_rate

    return cast(float, (win_rate * payout_ratio) - (loss_rate * 1.0))
