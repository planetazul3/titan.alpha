
import torch
import logging
from config.settings import load_settings
from models.core import DerivOmniModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = load_settings()
    logger.info("Initializing tiny model with settings...")
    
    device = settings.get_device()
    model = DerivOmniModel(settings).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {param_count:,} parameters")
    
    save_path = "checkpoints/best_model.pt"
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'config': settings.model_dump(),
        'global_step': 0,
        'best_val_loss': float('inf')
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved tiny model checkpoint to {save_path}")

if __name__ == "__main__":
    main()
