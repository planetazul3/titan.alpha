
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
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    args = parser.parse_args()

    settings = get_settings()
    
    # Load dataset
    logger.info(f"Loading data from {args.data_path}")
    dataset = DerivDataset(args.data_path)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DerivOmniModel(config=settings.model)
    
    # Configure trainer
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
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
