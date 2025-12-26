"""
Training orchestration for DerivOmniModel.

Module provides comprehensive training loop with checkpointing, early stopping,
learning rate scheduling, and metric tracking via TensorBoard.

Example:
    >>> from training import Trainer, TrainerConfig
    >>> trainer = Trainer(model, train_loader, val_loader, config, device)
    >>> trainer.train()
"""

from training.trainer import Trainer, TrainerConfig
from training.rl_trainer import RLTrainer
from training.online_learning import OnlineLearningModule

__all__ = [
    "Trainer",
    "TrainerConfig",
    "RLTrainer",
    "OnlineLearningModule",
]

