import torch
from typing import cast
import torch.nn as nn

from config.settings import Settings
from models.attention import AdditiveAttention
from models.blocks import BiLSTMBlock, MLPBlock
from models.tft import TemporalFusionTransformer


class TemporalExpert(nn.Module):
    """
    Temporal expert for directional prediction (Rise/Fall contracts).
    Architecture: BiLSTM/TFT -> Attention -> MLP -> Embedding
    """

    def __init__(self, settings: Settings, embedding_dim: int = 64, use_tft: bool = True, static_dim: int = 0):
        super().__init__()

        input_size = settings.data_shapes.feature_dim_candles
        hidden_size = settings.hyperparams.lstm_hidden_size
        self.use_tft = use_tft
        self.static_dim = static_dim

        if self.use_tft:
            self.tft = TemporalFusionTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                num_heads=4,  # Standard default
                dropout=settings.hyperparams.dropout_rate,
                static_input_size=static_dim,
            )
        # TFT outputs hidden_size dimensions
            encoder_out_dim = hidden_size
            
            # Audit Fix: Support Attention Pooling for TFT
            self.pooling_method = getattr(settings.hyperparams, "tft_pooling_method", "attention")
            if self.pooling_method == "attention":
                self.attention = AdditiveAttention(hidden_dim=encoder_out_dim)
            
        else:
            self.lstm = BiLSTMBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=settings.hyperparams.dropout_rate,
            )
            # BiLSTM outputs hidden_size * 2
            encoder_out_dim = hidden_size * 2
            
            self.attention = AdditiveAttention(hidden_dim=encoder_out_dim)

        self.projector = MLPBlock(
            layer_sizes=[encoder_out_dim, 128, embedding_dim],
            activation="silu",
            dropout=settings.hyperparams.dropout_rate,
        )

    def forward(self, candles: torch.Tensor, mask: torch.Tensor | None = None, static_context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            candles: (batch, seq_len, features) normalized candle data
            mask: (batch, seq_len) attention mask (optional)
            static_context: (batch, static_dim) static covariates (optional)
        Returns:
            embedding: (batch, embedding_dim) temporal representation
        """
        if self.use_tft:
            # TFT returns (output, attention_weights, feature_weights)
            # output: (batch, seq_len, hidden_size)
            tft_out, _, _ = self.tft(candles, mask=mask, static_covariates=static_context)
            
            if self.pooling_method == "attention":
                # Attention Pooling: Weighted sum of all time steps
                # Focuses on most relevant moments in history
                context, _ = self.attention(tft_out, mask=mask)
            elif self.pooling_method == "mean":
                # Mean Pooling: Average of all time steps (respecting mask)
                if mask is not None:
                     # mask: (batch, seq_len) -> (batch, seq_len, 1)
                     mask_expanded = mask.unsqueeze(-1).float()
                     tft_out = tft_out * mask_expanded
                     context = tft_out.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
                else:
                    context = tft_out.mean(dim=1)
            else:
                # Last Step: Standard sequence modeling (default "future-looking")
                # (batch, hidden_size)
                context = tft_out[:, -1, :]
        else:
            # lstm_out: (batch, seq_len, hidden*2)
            lstm_out, _ = self.lstm(candles)

            # context: (batch, hidden*2)
            context, _ = self.attention(lstm_out)

        # embedding: (batch, embedding_dim)
        embedding = self.projector(context)

        return cast(torch.Tensor, embedding)
