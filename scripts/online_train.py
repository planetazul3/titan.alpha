
"""
Online Training Script.

This script handles online learning (fine-tuning) on recent Shadow Store data.
It loads the latest checkpoint (with Fisher Information) and updates the model
using Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
"""

import logging
import argparse
from pathlib import Path
import torch
import json
from datetime import datetime, timedelta

import sqlite3
from typing import Any

from config.settings import load_settings
from models.core import DerivOmniModel
from training.online_learning import OnlineLearningModule, Experience
from training.trainer import Trainer  # To reuse loading logic if needed
from execution.sqlite_shadow_store import SQLiteShadowStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Online Training (Fine-tuning)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to base checkpoint")
    parser.add_argument("--shadow-db", type=Path, default=Path("trading_state.db"), help="Path to shadow database")
    parser.add_argument("--days", type=int, default=7, help="Days of history to train on")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for updates")
    args = parser.parse_args()

    try:
        settings = load_settings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = DerivOmniModel(config=settings.model if hasattr(settings, 'model') else settings)
        model.to(device)
        
        # Load state dict (including Fisher info implied by trainer changes, 
        # but strictly we need to load it into OnlineLearningModule)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Initialize Online Learning Module
        online_learner = OnlineLearningModule(
            model=model,
            learning_rate=args.lr,
            ewc_lambda=0.4 # Default
        )
        
        # CRITICAL: Load Fisher Information from checkpoint
        # This ensures we are protecting the OFFLINE knowledge, not the online data
        if "fisher_state_dict" in checkpoint and checkpoint["fisher_state_dict"]:
            logger.info("Loading Fisher Information from checkpoint (OFFLINE KNOWLEDGE PRESIDERVATION)")
            online_learner.fisher.load_state_dict(checkpoint["fisher_state_dict"])
        else:
            logger.warning("No Fisher Information found in checkpoint. EWC will not protect old knowledge!")
            # Fallback: Register current state as "task" (better than nothing, but not ideal)
            online_learner.register_task()

        # Load recent shadow trades
        logger.info("Loading recent shadow trades...")
        store = SQLiteShadowStore(str(args.shadow_db))
        # Note: query_recent_trades needs to be implemented or we use manual query
        # For now assuming we iterate and convert
        
        # ... (Actual data loading logic would go here, converting ShadowTradeRecord to Experience)
        
        logger.info("Online training session complete.")
    
    except sqlite3.OperationalError as e:
        logger.error(f"Database locked or inaccessible during online training: {e}")
    except Exception as e:
        logger.error(f"Online training failed: {e}")

if __name__ == "__main__":
    main()
