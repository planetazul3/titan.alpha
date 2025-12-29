"""
Centralized bootstrapping for the DerivOmniModel trading stack.

This module provides a unified factory for initializing core components,
ensuring consistent setup across live trading, backtesting, and production replays.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from config.settings import Settings
from data.features import get_feature_builder, FeatureBuilder
from data.ingestion.client import DerivClient
from execution.decision import DecisionEngine
from execution.regime import RegimeVeto
from execution.sqlite_shadow_store import SQLiteShadowStore
from models.core import DerivOmniModel
from tools.verify_checkpoint import verify_checkpoint

logger = logging.getLogger(__name__)


def create_trading_stack(
    settings: Settings,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    verify_ckpt: bool = True,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Initialize the complete trading stack.

    Args:
        settings: Loaded configuration settings.
        checkpoint_path: Path to model checkpoint (optional).
        device: 'cpu' or 'cuda' (overrides settings if provided).
        verify_ckpt: Whether to verify checkpoint integrity.
        client: Optional client instance (e.g. BacktestClient). If None, creates DerivClient.

    Returns:
        Dict containing:
        - 'model': Initialized DerivOmniModel
        - 'client': DerivClient
        - 'engine': DecisionEngine
        - 'shadow_store': SQLiteShadowStore
        - 'feature_builder': FeatureBuilder
        - 'regime_veto': RegimeVeto
    """
    # 1. Device Setup
    if not device:
        device = settings.get_device()
    logger.info(f"Initializing stack on device: {device}")

    # 2. Model Initialization
    logger.info("Initializing neural network model...")
    model = DerivOmniModel(settings).to(device)
    model.eval()

    model_version = "unversioned"

    # 3. Checkpoint Loading
    if checkpoint_path and checkpoint_path.exists():
        if verify_ckpt:
            logger.info(f"Verifying checkpoint: {checkpoint_path}")
            if not verify_checkpoint(checkpoint_path):
                raise RuntimeError(f"Checkpoint verification failed: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        try:
            # Inspection for architecture (BiLSTM vs TFT) happens inside model or here?
            # Ideally model handles it, but currently live.py does it. 
            # adhering to live.py logic:
            try:
                 checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception:
                 logger.warning("Checkpoint requires pickling allowed (weights_only=False)")
                 checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) # nosec

            # Check architecture
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            has_tft = any("temporal.tft" in k for k in state_dict.keys())
            
            if has_tft != settings.hyperparams.use_tft:
                 logger.warning(
                     f"Architecture mismatch! Checkpoint has_tft={has_tft}, "
                     f"Settings use_tft={settings.hyperparams.use_tft}. "
                     "Updating settings to match checkpoint."
                 )
                 settings.hyperparams.use_tft = has_tft
                 # Re-init model with correct architecture
                 model = DerivOmniModel(settings).to(device)
                 model.eval()

            model.load_state_dict(state_dict, strict=False)
            
            if "manifest" in checkpoint:
                manifest = checkpoint["manifest"]
                model_version = manifest.get("model_version", "unknown")
                logger.info(f"Loaded model version: {model_version}")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    else:
        logger.warning("No checkpoint provided. Model initialized with RANDOM weights.")

    # 4. Data & Feature Pipeline
    feature_builder = get_feature_builder(settings)
    
    # 5. Connectors & Stores
    if client is None:
        client = DerivClient(settings)
    
    shadow_store = SQLiteShadowStore(Path(settings.system.system_db_path))
    
    # 6. Safety & Decision
    regime_veto = RegimeVeto(
        threshold_caution=settings.hyperparams.regime_caution_threshold,
        threshold_veto=settings.hyperparams.regime_veto_threshold,
    )

    engine = DecisionEngine(
        settings, 
        regime_veto=regime_veto, 
        shadow_store=shadow_store, 
        model_version=model_version
    )

    return {
        "model": model,
        "client": client,
        "engine": engine,
        "shadow_store": shadow_store,
        "feature_builder": feature_builder,
        "regime_veto": regime_veto,
        "device": device
    }
