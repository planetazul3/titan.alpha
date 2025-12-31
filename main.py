import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from execution.sqlite_shadow_store import SQLiteShadowStore  # Full context capture
    from execution.safety_store import SQLiteSafetyStateStore
    from execution.idempotency_store import SQLiteIdempotencyStore

    from config.settings import load_settings
    from execution.decision import DecisionEngine
    from models.core import DerivOmniModel
    from utils.seed import set_global_seed
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)


async def main():
    logger.info("Starting Unified Trading System...")

    # 1. Load Settings
    try:
        settings = load_settings()
        logger.info(f"Settings loaded. Environment: {settings.environment}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return

    # 2. Set Seed
    set_global_seed(settings.seed)
    logger.info(f"Global seed set to {settings.seed}")

    # 3. Initialize Model
    device = settings.get_device()
    logger.info(f"Using device: {device}")

    try:
        model = DerivOmniModel(settings).to(device)
        model.eval()
        logger.info("DerivOmniModel initialized successfully")
        logger.info(f"Model parameters: {model.count_parameters():,}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Initialize Safety and Idempotency Stores
    shadow_store = SQLiteShadowStore(Path("data_cache/trading_state.db"))
    safety_store = SQLiteSafetyStateStore(Path("data_cache/trading_state.db"))
    idempotency_store = SQLiteIdempotencyStore(Path("data_cache/idempotency.db"))
    
    # 5. Initialize Decision Engine with full safety context
    engine = DecisionEngine(
        settings, 
        shadow_store=shadow_store,
        safety_store=safety_store
    )
    logger.info("DecisionEngine initialized with safety and idempotency protection")

    # 6. Simulate Run (Mock Data)
    logger.info("Generating mock data for simulation...")

    batch_size = 2  # Simulate 2 samples
    seq_ticks = settings.data_shapes.sequence_length_ticks
    seq_candles = settings.data_shapes.sequence_length_candles

    # Mock Tensors
    # Ticks: (batch, seq_len) -> SpatialExpert expects (batch, 1, seq_len) inside
    # Wait, SpatialExpert forward checks dim.
    # core.py forward passes `ticks` directly to spatial.
    # dataset.py produces `ticks` as (seq_len,). Collated batch: (batch, seq_len).
    # SpatialExpert handles unsqueeze.

    ticks = torch.randn(batch_size, seq_ticks).to(device)

    # Candles: (batch, seq_len, 10)
    # TemporalExpert input_size=10 hardcoded in temporal.py
    candles = torch.randn(batch_size, seq_candles, 10).to(device)

    # Vol Metrics: (batch, 4)
    vol_metrics = torch.randn(batch_size, 4).to(device)

    logger.info(
        f"Input shapes: Ticks={ticks.shape}, Candles={candles.shape}, Vol={vol_metrics.shape}"
    )

    # 7. Forward Pass
    try:
        with torch.no_grad():
            # Offload heavy model inference to a separate thread to keep the event loop responsive
            probs = await asyncio.to_thread(model.predict_probs, ticks, candles, vol_metrics)

        logger.info("Model prediction successful.")
        for k, v in probs.items():
            logger.info(f"Output {k}: {v.detach().cpu().numpy().tolist()}")

    except Exception as e:
        logger.error(f"Runtime error during model execution: {e}")
        import traceback

        traceback.print_exc()
        return

    # 8. Decision Processing
    # Convert tensor output to list of dicts for engine
    # Engine expects Dict[str, float] for a single timestamp usually, or we loop

    logger.info("Processing decisions...")
    for i in range(batch_size):
        # Extract scalar probabilities for current sample
        sample_probs = {}
        for k, v in probs.items():
            if k == "vol_reconstruction":
                # Reconstruction error is (batch,) or (batch, 4) depending on how it's returned
                # It's usually a scalar per sample for anomaly score, but let's check
                # VolatilityExpert.reconstruct returns (batch, input_dim) - it's the reconstructed vector
                # But predict_probs returns it directly.
                # We shouldn't use reconstructed vector as probability.
                continue
            sample_probs[k] = v[i].item()

        logger.info(f"Sample {i} probabilities: {sample_probs}")

        real_trades = await engine.process_model_output(
            sample_probs,
            timestamp=datetime.now(timezone.utc),
            market_data={"simulated": True, "sample_id": i},
            reconstruction_error=0.1,  # Mock value for demo
        )

        if real_trades:
            logger.info(f"  -> Generated {len(real_trades)} REAL trades!")
            for t in real_trades:
                logger.info(f"     {t}")
        else:
            logger.info("  -> No real trades generated (filtered or shadow).")

    # Check Shadow Log
    stats = engine.get_statistics()
    logger.info(f"Session Statistics: {stats}")

    # 8. Graceful Shutdown
    await engine.shutdown()
    logger.info("System execution test complete.")


if __name__ == "__main__":
    asyncio.run(main())
