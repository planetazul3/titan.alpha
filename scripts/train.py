
"""
Offline Training Script.

This script manages the offline training of the DerivOmniModel.
It uses the Trainer class to orchestrate the training loop,
handles data loading, and saves the final checkpoint with
Fisher Information for Elastic Weight Consolidation (EWC).
"""

import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from config.settings import get_settings
from models.core import DerivOmniModel
from data.dataset import DerivDataset  # Assuming this exists
from training.trainer import Trainer, TrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train DerivOmniModel")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to training data (parquet)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides settings)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides settings)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides settings)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    args = parser.parse_args()

    settings = get_settings()
    
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
    # dataset.candle_len is the sequence length
    gap = settings.data_shapes.sequence_length_candles + dataset.lookahead
    
    val_start = train_size + gap
    
    if val_start >= total_len:
        logger.warning("Dataset too small for proper temporal split with gap! Reducing gap/train size.")
        val_start = train_size # Fallback (risky but better than crashing)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(val_start, total_len))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Temporal Split: Train={len(train_dataset)}, Val={len(val_dataset)} (Gap={gap})")
    
    # Use settings for hyperparameters unless overridden
    batch_size = args.batch_size or settings.hyperparams.batch_size
    lr = args.lr or settings.hyperparams.learning_rate
    epochs = args.epochs or 50 # Default if not in settings (settings doesn't have epochs)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle WITHIN the training set is fine
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = DerivOmniModel(config=settings)
    
    # Configure trainer
    config = TrainerConfig(
        epochs=epochs,
        learning_rate=lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Run training
    # This will now compute and save Fisher Information at the end
    metrics = trainer.train()
    
    logger.info(f"Training completed. Final Val Loss: {metrics['final_val_loss']:.4f}")

if __name__ == "__main__":
    main()
