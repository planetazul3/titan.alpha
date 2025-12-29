
"""
Offline Training Script.

This script manages the offline training of the DerivOmniModel.
It uses the Trainer class to orchestrate the training loop,
handles data loading, and saves the final checkpoint with
Fisher Information for Elastic Weight Consolidation (EWC).
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split

from config.settings import load_settings
from models.core import DerivOmniModel
from data.dataset import DerivDataset  
from training.trainer import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"RNG seeds set to {seed}")

def main():
    parser = argparse.ArgumentParser(description="Train DerivOmniModel")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to training data (parquet)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides settings)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides settings)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides settings)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    settings = load_settings()
    
    # Load dataset
    logger.info(f"Loading data from {args.data_path}")
    dataset = DerivDataset(args.data_path, settings)
    
    # Temporal Split (Fix Data Leakage)
    # We must ensure no overlap between train and val sequences
    # Purge gap = candle_len + lookahead
    # (Actually, just lookahead is strictly required if we want to predict FUTURE, 
    # but considering sequence overlap, a gap of sequence_length is safer to ensure 
    # absolutely no information leakage from future to past in the validation set)
    
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    
    # Gap to ensure no overlap in sliding windows
    # Must account for: sequence_length + warmup_steps + lookahead
    gap = settings.data_shapes.sequence_length_candles + settings.data_shapes.warmup_steps + dataset.lookahead
    
    val_start = train_size + gap
    
    if val_start >= total_len:
        # STRICT: Fail instead of falling back to risky split that could cause data leakage
        min_required = int(gap / 0.2) + 1  # Minimum dataset size for 80/20 split with gap
        raise ValueError(
            f"Dataset too small for proper temporal split with gap! "
            f"Have {total_len} samples, need at least {min_required}. "
            f"Gap = {gap} (seq_len={settings.data_shapes.sequence_length_candles} + "
            f"warmup={settings.data_shapes.warmup_steps} + lookahead={dataset.lookahead}). "
            f"Either increase dataset size or reduce sequence_length/warmup_steps."
        )
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(val_start, total_len))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Temporal Split: Train={len(train_dataset)}, Val={len(val_dataset)} (Gap={gap})")
    
    # Use settings for hyperparameters unless overridden
    batch_size = args.batch_size or settings.hyperparams.batch_size
    lr = args.lr or settings.hyperparams.learning_rate
    epochs = args.epochs or 50 # Default if not in settings (settings doesn't have epochs)
    
    # Dynamic worker count for Kaggle/Cloud environments
    num_workers = os.cpu_count() or 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle WITHIN the training set is fine
        drop_last=True,  # Ensure consistent batch sizes
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model
    model = DerivOmniModel(settings)
    
    # Configure trainer
    config = TrainerConfig(
        epochs=epochs,
        learning_rate=lr,
        checkpoint_dir=args.checkpoint_dir,
        ewc_sample_size=settings.hyperparams.ewc_sample_size # Added this line
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        settings=settings
    )
    
    # Run training
    # This will now compute and save Fisher Information at the end
    metrics = trainer.train()
    
    logger.info(f"Training completed. Final Val Loss: {metrics['final_val_loss']:.4f}")

if __name__ == "__main__":
    main()
