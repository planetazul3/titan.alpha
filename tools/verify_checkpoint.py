#!/usr/bin/env python3
"""
Checkpoint Verification Utility (x.titan).
Performs smoke tests on model checkpoints before deployment.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from models.core import DerivOmniModel
from data.features import FeatureBuilder

def verify_checkpoint(checkpoint_path: Path):
    """
    Perform smoke tests on a model checkpoint.
    """
    print(f"--- Verifying Checkpoint: {checkpoint_path.name} ---")
    
    # 1. Load Checkpoint
    try:
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            return False
            
        print("1. Loading checkpoint...")
        # Load with weights_only=True first (safer)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            print("   WARNING: Failed with weights_only=True. Trying legacy load (risky if untrusted).")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
        if "model_state_dict" not in checkpoint:
            print("   ERROR: 'model_state_dict' not found in checkpoint.")
            return False
        print("   SUCCESS: Checkpoint loaded.")
    except Exception as e:
        print(f"   ERROR: Failed to load checkpoint: {e}")
        return False

    # 2. Inspect Architecture
    try:
        print("2. Inspecting architecture...")
        state_dict = checkpoint["model_state_dict"]
        has_tft = any("temporal.tft" in k for k in state_dict.keys())
        arch_type = "TFT" if has_tft else "BiLSTM"
        print(f"   Detected architecture: {arch_type}")
        
        # Load settings and set use_tft accordingly
        settings = load_settings()
        settings.hyperparams.use_tft = has_tft
    except Exception as e:
        print(f"   ERROR: Architecture inspection failed: {e}")
        return False

    # 3. Model Instantiation
    try:
        print("3. Instantiating model...")
        model = DerivOmniModel(settings)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"   SUCCESS: Model instantiated ({model.count_parameters():,} parameters)")
    except Exception as e:
        print(f"   ERROR: Model instantiation failed: {e}")
        return False

    # 4. Smoke Test (Inference)
    try:
        print("4. Running smoke test (dummy inference)...")
        # Create dummy input data
        tick_len = settings.data_shapes.sequence_length_ticks + settings.data_shapes.warmup_steps
        candle_len = settings.data_shapes.sequence_length_candles + settings.data_shapes.warmup_steps
        
        # Create dummy tick and candle arrays (HITS: Must be positive for log returns)
        base_price = 100.0
        dummy_ticks = (base_price + np.random.randn(tick_len) * 5).astype(np.float32)
        dummy_candles = np.zeros((candle_len, 6), dtype=np.float32)
        for i in range(candle_len):
             c_base = base_price + np.random.randn() * 5
             dummy_candles[i] = [
                 c_base, # open
                 c_base + abs(np.random.randn()), # high
                 c_base - abs(np.random.randn()), # low
                 c_base + np.random.randn(), # close
                 100.0, # volume
                 1600000000.0 + i * 60 # timestamp
             ]
        
        # Build features
        fb = FeatureBuilder(settings)
        features = fb.build(ticks=dummy_ticks, candles=dummy_candles)
        
        t_tensor = features["ticks"].unsqueeze(0)
        c_tensor = features["candles"].unsqueeze(0)
        v_tensor = features["vol_metrics"].unsqueeze(0)
        
        # Run prediction
        with torch.no_grad():
            probs = model.predict_probs(t_tensor, c_tensor, v_tensor)
            recon_error = model.get_volatility_anomaly_score(v_tensor).item()
            
        print(f"   SUCCESS: Inference completed.")
        print(f"   Outputs: { {k: f'{v.item():.4f}' for k, v in probs.items()} }")
        print(f"   Reconstruction Error: {recon_error:.4f}")
        
        # Validate outputs
        for k, v in probs.items():
            if not (0 <= v.item() <= 1):
                print(f"   ERROR: Probability {k}={v.item():.4f} is out of [0, 1] range!")
                return False
        
        if recon_error < 0:
            print(f"   ERROR: Reconstruction error {recon_error:.4f} is negative!")
            return False
            
    except Exception as e:
        print(f"   ERROR: Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- CHECKPOINT VERIFICATION PASSED ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a model checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()
    
    success = verify_checkpoint(Path(args.checkpoint))
    sys.exit(0 if success else 1)
