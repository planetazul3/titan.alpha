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
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.constants import MAX_CONSECUTIVE_NANS
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
    early_stop_min_delta: float = 1e-4
    gradient_clip: float = 1.0

    # Advanced training options
    use_amp: bool = True
    use_uncertainty_loss: bool = True  # Learned loss weights
    gradient_accumulation_steps: int = 1
    
    # EWC
    ewc_sample_size: int = 2000


class BatchProfiler:
    """Simple profiler for batch timing."""
    def __init__(self):
        self.data_time = 0.0
        self.compute_time = 0.0
        self.last_step = time.time()
        
    def step_data(self):
        """Call after data loading finishes."""
        now = time.time()
        self.data_time += now - self.last_step
        self.last_step = now
        
    def step_compute(self):
        """Call after compute finishes."""
        now = time.time()
        self.compute_time += now - self.last_step
        self.last_step = now
        
    def reset(self):
        self.data_time = 0.0
        self.compute_time = 0.0
        self.last_step = time.time()


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
        settings: Any = None,  # Pass full settings for advanced config
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.settings = settings  # Store settings for manifest saving
        
        # Optimize model with torch.compile if available and not skipped (PyTorch 2.0+)
        # We check for the attribute first to be safe
        if hasattr(torch, "compile") and device and device.type == "cuda":
            logger.info("Compiling model with torch.compile for performance...")
            try:
                # Type cast to satisfy mypy assignment check
                self.model = cast(nn.Module, torch.compile(self.model)) 
            except Exception as e:
                logger.warning(f"Model compilation failed, falling back to eager mode: {e}")
                
        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        # Multi-task loss
        self.criterion = MultiTaskLoss().to(device)

        # Optimizer (include model and criterion parameters if learnable)
        trainable_params = list(model.parameters())
        if config.use_uncertainty_loss:
            trainable_params += list(self.criterion.parameters())

        self.optimizer = AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # Learning rate scheduler with proper warmup (S01)
        # 1. Linear warmup from lr/10 to lr
        warmup_steps = config.warmup_epochs
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            )
            # 2. Cosine annealing from lr to eta_min
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - warmup_steps, eta_min=1e-6
            )
            # Combine them
            # Cast to Any to bypass strict type check for now since SequentialLR behaves as LRScheduler
            self.scheduler: Any = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )

        # Mixed Precision Training setup
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.autocast_dtype = torch.float16 if self.use_amp else torch.float32

        # Gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps

        # Metrics trackers for multiple tasks
        self.task_names = ["rise_fall", "touch", "range"]
        self.task_metrics = {name: TradingMetrics() for name in self.task_names}
        
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
        self._consecutive_nan_count = 0  # Track consecutive NaN losses
        self._max_consecutive_nans = MAX_CONSECUTIVE_NANS

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

            # Learning rate scheduling
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
                
                # Log uncertainty weights if used
                if self.config.use_uncertainty_loss:
                    for i, name in enumerate(["rise_fall", "touch", "range", "reconstruction"]):
                        sigma = torch.exp(self.criterion.log_vars[i] * 0.5).item()
                        self.writer.add_scalar(f"Uncertainty/sigma_{name}", sigma, epoch)

                # Log all task metrics
                for name, metrics in val_metrics.items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"Metrics_{name}/{metric_name}", value, epoch)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Best model saving with min_delta (ES01)
            if val_loss < self.best_val_loss - self.config.early_stop_min_delta:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
                self.patience_counter = 0
                logger.info(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Memory Cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

        # Restore best weights (ES02)
        best_path = self.config.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            logger.info("Restoring best model weights for final evaluation...")
            self.load_checkpoint(best_path)

        # Compute Fisher Information on training data for EWC
        logger.info("Computing Fisher Information for EWC...")
        self.fisher_info = FisherInformation(self.model)
        # Use a subset of training data for efficiency
        self.fisher_info.compute(
            self.train_loader, 
            self.criterion, 
            num_samples=self.config.ewc_sample_size
        )

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
        
        # Batch Profiler
        profiler = BatchProfiler()

        n_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Record Data Time
            profiler.step_data()
            
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

            # CRITICAL: Check for NaN/Inf loss with enhanced monitoring
            if not torch.isfinite(loss):
                self._consecutive_nan_count += 1
                # Log scaler state for debugging AMP issues
                scaler_scale = self.scaler.get_scale() if self.use_amp else 1.0
                logger.warning(
                    f"Loss is {loss.item()} (non-finite) at step {self.global_step}. "
                    f"Consecutive NaN count: {self._consecutive_nan_count}, "
                    f"Scaler scale: {scaler_scale:.2e}"
                )
                if self.writer:
                    self.writer.add_scalar("Debug/nan_count", self._consecutive_nan_count, self.global_step)
                    self.writer.add_scalar("Debug/scaler_scale", scaler_scale, self.global_step)
                
                # Halt training if too many consecutive NaNs
                if self._consecutive_nan_count >= self._max_consecutive_nans:
                    raise RuntimeError(
                        f"Training halted: {self._consecutive_nan_count} consecutive NaN losses. "
                        f"Check learning rate, data, or reduce AMP precision."
                    )
                
                self.optimizer.zero_grad(set_to_none=True)
                # Log that we skipped this batch
                if self.writer:
                    self.writer.add_scalar("Debug/batch_skipped", 1, self.global_step)
                continue
            
            # Reset NaN counter on successful finite loss
            self._consecutive_nan_count = 0

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps OR on the last batch
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == n_batches
            
            if is_accumulation_step or is_last_batch:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Log grad norm to TB
                if self.writer and self.global_step % 10 == 0:
                    self.writer.add_scalar("Grad/norm", grad_norm, self.global_step)

                # Reset gradients
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            # Track loss (unscaled for logging)
            current_loss = loss.item() * (
                self.accumulation_steps if self.accumulation_steps > 1 else 1
            )
            total_loss += current_loss
            num_batches += 1

            # Record Compute Time
            profiler.step_compute()
            
            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{current_loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
            )
            
            # Log profiling stats periodically
            if self.writer and self.global_step % 100 == 0:
                 self.writer.add_scalar("Time/data", profiler.data_time, self.global_step)
                 self.writer.add_scalar("Time/compute", profiler.compute_time, self.global_step)
            
            # Reset profiler for next batch
            profiler.reset()

        return total_loss / num_batches

    def _validate_epoch(self) -> tuple[float, dict[str, dict[str, float]]]:
        """Run validation epoch with multi-task metrics."""
        self.model.eval()
        for m in self.task_metrics.values():
            m.reset()

        if len(self.val_loader) == 0:
            return 0.0, {}

        val_loss_sum = 0.0
        n_valid_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Move to device and process
            ticks = batch["ticks"].to(self.device, non_blocking=True)
            candles = batch["candles"].to(self.device, non_blocking=True)
            vol_metrics = batch["vol_metrics"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}

            # Forward pass with autocast for consistency
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp
            ):
                with torch.no_grad():
                    outputs = self.model(ticks, candles, vol_metrics)
                    losses = self.criterion(outputs, targets, vol_metrics)

            loss_val = losses["total"].item()
            if not torch.isfinite(torch.tensor(loss_val)):
                logger.warning("Validation batch yielded non-finite loss! Skipping for total.")
                continue

            val_loss_sum += loss_val
            n_valid_batches += 1

            # Track predictions for ALL metrics
            for name in self.task_names:
                logit = outputs.get(f"{name}_logit")
                target = targets.get(name)
                if logit is not None and target is not None:
                    prob = torch.sigmoid(logit.float())
                    self.task_metrics[name].update(prob, target)

        val_loss = val_loss_sum / n_valid_batches if n_valid_batches > 0 else float("inf")
        
        # Compute metrics for each task
        all_metrics = {}
        for name, tracker in self.task_metrics.items():
            all_metrics[name] = tracker.compute()

        return val_loss, all_metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint with full training state (Atomic)."""
        import os
        path = self.config.checkpoint_dir / filename
        temp_path = path.with_suffix(".tmp")
        
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),  # Save AMP scaler state
            "fisher_state_dict": self.fisher_info.state_dict() if self.fisher_info else None,
            "best_val_loss": self.best_val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Issue 10: Add manifest for validation
        if self.settings:
            from config.constants import FEATURE_SCHEMA_VERSION
            state["manifest"] = {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "data_shapes": self.settings.data_shapes.model_dump(),
                "model_version": getattr(self.settings.system, "model_version", "1.0.0"),
                "timestamp": time.time(),
            }
        
        # Atomic write
        torch.save(state, temp_path)
        os.replace(temp_path, path)
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
