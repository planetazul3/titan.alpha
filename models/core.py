"""
Core neural network architecture for DerivOmniModel.

ARCHITECTURE PRINCIPLE: MODEL = EMBEDDINGS + PROBABILITIES ONLY
This model outputs embeddings and probabilities. It does NOT make trading decisions.
All decision logic (thresholding, filtering, regime veto) lives in execution/decision.py.

This separation ensures:
- Model is pure inference (deterministic given inputs)
- Policy changes don't require model retraining
- Easy A/B testing of different decision policies

Model outputs:
- Embeddings: Expert networks produce 64-dim embeddings
- Probabilities: predict_probs() outputs [0,1] probabilities per contract type
- Reconstruction error: get_volatility_anomaly_score() for regime detection

Decision logic lives in:
- execution/filters.py: classify_probability(), filter_signals()
- execution/decision.py: DecisionEngine, regime veto
- execution/regime.py: RegimeVeto

Example:
    >>> model = DerivOmniModel(settings)
    >>> probs = model.predict_probs(ticks, candles, vol_metrics)  # Pure inference
    >>> trades = decision_engine.process_model_output(probs)      # Decision logic
"""

import logging

import torch
import torch.nn as nn

from config.settings import Settings
from models.fusion import ExpertFusion
from models.heads import create_contract_heads
from models.spatial import SpatialExpert
from models.temporal import TemporalExpert
from models.volatility import VolatilityExpert

logger = logging.getLogger(__name__)


class DerivOmniModel(nn.Module):
    """
    Unified multi-expert model for Deriv binary options trading.

    Architecture:
    1. Three expert networks process different data modalities:
       - Temporal (candles) -> embedding_dim=64
       - Spatial (ticks) -> embedding_dim=64
       - Volatility (metrics) -> latent_dim from settings
    2. ExpertFusion combines embeddings -> 256-dim context
    3. Contract-specific heads predict outcomes

    Args:
        settings: Configuration containing hyperparameters and data shapes
    """

    def __init__(self, settings: Settings):
        super().__init__()

        logger.info("Initializing DerivOmniModel...")

        # Dimensions for embeddings
        temp_dim = settings.hyperparams.temporal_embed_dim
        spat_dim = settings.hyperparams.spatial_embed_dim
        vol_dim = settings.hyperparams.latent_dim
        fusion_out = settings.hyperparams.fusion_output_dim
        
        # Get usage flag (defaults to True if not in settings, but we added it)
        # Get usage flag (defaults to True if not in settings, but we added it)
        use_tft = getattr(settings.hyperparams, "use_tft", True)

        # M06: Pass volatility embedding as static context to TFT
        # This allows the temporal model to adapt to the volatility regime
        self.temporal = TemporalExpert(
            settings, 
            embedding_dim=temp_dim, 
            use_tft=use_tft,
            static_dim=vol_dim if use_tft else 0
        )
        self.spatial = SpatialExpert(settings, embedding_dim=spat_dim)

        # Volatility expert input dim is configurable via settings
        vol_input_dim = settings.data_shapes.feature_dim_volatility
        self.volatility = VolatilityExpert(input_dim=vol_input_dim, settings=settings, hidden_dim=32)

        self.fusion = ExpertFusion(
            temporal_dim=temp_dim,
            spatial_dim=spat_dim,
            volatility_dim=vol_dim,
            output_dim=fusion_out,
            dropout=settings.hyperparams.fusion_dropout,
        )

        self.heads = create_contract_heads(input_dim=fusion_out, settings=settings)

        total_params = self.count_parameters()
        logger.info(f"DerivOmniModel initialized with {total_params:,} parameters")

    def forward(
        self, 
        ticks: torch.Tensor, 
        candles: torch.Tensor, 
        vol_metrics: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through all experts and heads.

        Args:
            ticks: Tick sequence, shape (batch, seq_len_ticks)
            candles: Candle features, shape (batch, seq_len_candles, 10)
            vol_metrics: Volatility metrics, shape (batch, 4)
            masks: Optional dictionary of masks (e.g. 'candles_mask')

        Returns:
            Dictionary of logits:
                'rise_fall_logit': (batch, 1)
                'touch_logit': (batch, 1)
                'range_logit': (batch, 1)
        """
        candles_mask = masks.get("candles_mask") if masks else None
        ticks_mask = masks.get("ticks_mask") if masks else None
        
        emb_vol = self.volatility.encode(vol_metrics)  # (batch, vol_dim)
        
        # Pass volatility embedding as static context to TemporalExpert (if using TFT)
        emb_temp = self.temporal(candles, mask=candles_mask, static_context=emb_vol)  # (batch, temp_dim)
        emb_spat = self.spatial(ticks, mask=ticks_mask)  # (batch, spat_dim)

        context = self.fusion(emb_temp, emb_spat, emb_vol)  # (batch, fusion_out)

        logits = {
            "rise_fall_logit": self.heads["rise_fall"](context),
            "touch_logit": self.heads["touch"](context),
            "range_logit": self.heads["range"](context),
            "vol_reconstruction": self.volatility.reconstruct(vol_metrics),
        }

        return logits

    def predict_probs(
        self, 
        ticks: torch.Tensor, 
        candles: torch.Tensor, 
        vol_metrics: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with Sigmoid applied for inference.
        """
        logits = self(ticks, candles, vol_metrics, masks=masks)
        probs = {
            k.replace("_logit", "_prob"): torch.sigmoid(v) 
            for k, v in logits.items() 
            if k.endswith("_logit")
        }
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_volatility_anomaly_score(self, vol_metrics: torch.Tensor) -> torch.Tensor:
        """
        Get reconstruction error from volatility expert.
        """
        return self.volatility.reconstruction_error(vol_metrics)
