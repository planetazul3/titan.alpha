"""
Training loop for DerivOmniModel.

Supports:
- GPU/TPU training (Colab-ready)
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Checkpointing
- TensorBoard logging
- Early stopping
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.losses import MultiTaskLoss
from training.metrics import TradingMetrics
from training.online_learning import FisherInformation

logger = logging.getLogger(__name__)

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@dataclass
class TrainerConfig:
    """Configuration for trainer."""

    epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs/tensorboard"))
    save_every: int = 5
    early_stop_patience: int = 10
    gradient_clip: float = 1.0

    # Advanced training options (backward compatible defaults)
    use_amp: bool = True  # Enable Automatic Mixed Precision on CUDA
    gradient_accumulation_steps: int = 1  # 1 = no accumulation


class Trainer:
    """
    Training orchestrator for DerivOmniModel.

    Features:
    - Mixed Precision Training (AMP) for faster GPU training
    - Gradient Accumulation for larger effective batch sizes
    - Gradient clipping for training stability
    - Checkpointing with full state (model, optimizer, scheduler, scaler)
    - TensorBoard logging
    - Early stopping

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        # Loss and optimizer
        self.criterion = MultiTaskLoss()
        self.optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6
        )

        # Mixed Precision Training setup
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.autocast_dtype = torch.float16 if self.use_amp else torch.float32

        # Gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps

        # Metrics tracker
        self.metrics = TradingMetrics()
        
        # EWC State
        self.fisher_info: FisherInformation | None = None

        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            config.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(config.log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log configuration
        logger.info("Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed Precision (AMP): {self.use_amp}")
        logger.info(f"  Gradient Accumulation Steps: {self.accumulation_steps}")

    def _verify_gpu_usage(self):
        """Verify GPU is actually being used with a test operation."""
        if self.device.type == "cuda":
            # Force CUDA initialization and verify
            test_tensor = torch.randn(1000, 1000, device=self.device)
            result = torch.mm(test_tensor, test_tensor)
            torch.cuda.synchronize()  # Wait for GPU to complete
            del test_tensor, result
            torch.cuda.empty_cache()

            # Log GPU memory to confirm usage
            allocated = torch.cuda.memory_allocated() / 1e6
            logger.info(f"GPU verification passed. Memory allocated: {allocated:.1f} MB")
            return True
        return False

    def train(self) -> dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dict with final metrics and best checkpoint path
        """
        # Verify GPU is working before training
        print(f"\n{'=' * 60}", flush=True)
        print("🚀 TRAINING STARTING", flush=True)
        print(f"   Device: {self.device}", flush=True)
        print(f"   AMP enabled: {self.use_amp}", flush=True)
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_name}", flush=True)
            self._verify_gpu_usage()
            print("   ✅ GPU verification passed!", flush=True)
        else:
            print("   ⚠️  Training on CPU (slow!)", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss, val_metrics = self._validate_epoch()

            # Learning rate scheduling (after warmup)
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, LR: {lr:.2e}"
            )

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("LR", lr, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"Metrics/{k}", v, epoch)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Best model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

        # Compute Fisher Information on training data for EWC
        logger.info("Computing Fisher Information for EWC...")
        self.fisher_info = FisherInformation(self.model)
        # Use a subset of training data for efficiency (e.g., 2000 samples)
        self.fisher_info.compute(self.train_loader, self.criterion, num_samples=2000)

        # Save final model with Fisher info
        self._save_checkpoint("final_model.pt")

        if self.writer:
            self.writer.close()

        return {
            "final_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.current_epoch + 1,
            "best_checkpoint": self.config.checkpoint_dir / "best_model.pt",
        }

    def _train_epoch(self) -> float:
        """Run single training epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Reset gradients at start
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device with non-blocking transfer for async
            ticks = batch["ticks"].to(self.device, non_blocking=True)
            candles = batch["candles"].to(self.device, non_blocking=True)
            vol_metrics = batch["vol_metrics"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}

            # Forward pass with autocast for mixed precision
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp
            ):
                outputs = self.model(ticks, candles, vol_metrics)
                losses = self.criterion(outputs, targets, vol_metrics)
                loss = losses["total"]

                # Scale loss for gradient accumulation
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Reset gradients
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            # Track loss (unscaled for logging)
            current_loss = loss.item() * (
                self.accumulation_steps if self.accumulation_steps > 1 else 1
            )
            total_loss += current_loss
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{current_loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
            )

        return total_loss / num_batches

    @torch.no_grad()
    def _validate_epoch(self) -> tuple:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Move to device with non-blocking transfer
            ticks = batch["ticks"].to(self.device, non_blocking=True)
            candles = batch["candles"].to(self.device, non_blocking=True)
            vol_metrics = batch["vol_metrics"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}

            # Forward pass with autocast for consistency
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp
            ):
                outputs = self.model(ticks, candles, vol_metrics)
                losses = self.criterion(outputs, targets, vol_metrics)

            total_loss += losses["total"].item()

            # Track predictions for metrics (use rise_fall as primary)
            rise_fall_logit = outputs.get("rise_fall_logit")
            rise_fall_target = targets.get("rise_fall")

            if rise_fall_logit is not None and rise_fall_target is not None:
                # Convert logit to probability for metrics
                rise_fall_prob = torch.sigmoid(rise_fall_logit.float())
                self.metrics.update(rise_fall_prob, rise_fall_target)

        val_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.compute()

        return val_loss, metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint with full training state."""
        path = self.config.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),  # Save AMP scaler state
                "fisher_state_dict": self.fisher_info.state_dict() if self.fisher_info else None,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model from checkpoint with full training state."""
        # Using weights_only=False as we load full training state (optimizer, etc.)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)  # nosec B614
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load AMP scaler state if available
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        # Load Fisher info if available
        if checkpoint.get("fisher_state_dict"):
            self.fisher_info = FisherInformation(self.model)
            self.fisher_info.load_state_dict(checkpoint["fisher_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
