#!/usr/bin/env python3
"""
Evaluation script for DerivOmniModel.
Loads best_model.pt and runs validation on the test/validation split.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from config.settings import load_settings
from data.dataset import DerivDataset
from models.core import DerivOmniModel
from training.losses import MultiTaskLoss
from training.metrics import TradingMetrics
from tqdm.auto import tqdm


from utils.logging_setup import setup_logging

logger, log_dir, log_file = setup_logging(script_name="evaluate")

def main(args):
    settings = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return 1

    dataset = DerivDataset(data_path, settings, mode="train")
    
    # Replicate train/val split from training to ensure we evaluate on unseen data
    # C02: Use temporal split to prevent data leakage
    # Replicate train/val split from training (temporal)
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    
    # C02: Purge gap logic
    purge_gap = settings.data_shapes.sequence_length_candles
    val_start_idx = train_size + purge_gap
    
    if val_start_idx >= total_len:
        logger.warning("Dataset too small for purge gap! Resizing split to avoid error.")
        val_start_idx = train_size
    
    val_indices = list(range(val_start_idx, total_len))
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Evaluating on {len(val_dataset)} validation samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Load Model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Pre-inspect checkpoint to determine architecture (BiLSTM vs TFT)
    logger.info(f"Inspecting checkpoint: {checkpoint_path.name}")
    try:
        # Load to CPU first to inspect structure
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Detect architecture type
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        has_tft = any("temporal.tft" in k for k in state_dict.keys())
        
        if has_tft:
            settings.hyperparams.use_tft = True
            logger.info("Detected TFT architecture in checkpoint")
        else:
            settings.hyperparams.use_tft = False
            logger.info("Detected BiLSTM architecture in checkpoint (Legacy)")
            
    except Exception as e:
        logger.error(f"Failed to inspect checkpoint: {e}")
        # Fall through - will try to load normally and might fail

    model = DerivOmniModel(settings).to(device)
    
    logger.info(f"Loading checkpoint weights: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # 3. Evaluation Loop
    criterion = MultiTaskLoss()
    metrics = TradingMetrics()
    total_loss = 0.0
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if args.limit and i >= args.limit:
                break
                
            ticks = batch["ticks"].to(device)
            candles = batch["candles"].to(device)
            vol_metrics = batch["vol_metrics"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            outputs = model(ticks, candles, vol_metrics)
            losses = criterion(outputs, targets, vol_metrics)
            total_loss += losses["total"].item()

            # Update metrics (Rise/Fall accuracy)
            rise_fall_logit = outputs.get("rise_fall_logit")
            rise_fall_target = targets.get("rise_fall")
            
            if rise_fall_logit is not None and rise_fall_target is not None:
                prob = torch.sigmoid(rise_fall_logit.float())
                metrics.update(prob, rise_fall_target)

    # 4. Report Results
    num_batches = len(val_loader)
    if args.limit:
        num_batches = min(num_batches, args.limit)
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    final_metrics = metrics.compute()
    
    print("\n" + "="*40)
    print(f"📊 EVALUATION RESULTS")
    print("="*40)
    print(f"Model: {checkpoint_path.name}")
    print(f"Epochs Trained: {checkpoint.get('epoch', '?')}")
    print(f"Loss: {avg_loss:.4f}")
    print("-" * 20)
    print("Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("="*40 + "\n")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--data", type=str, default="data_cache", help="Path to data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of batches to evaluate")
    
    args = parser.parse_args()
    sys.exit(main(args))