import torch
from typing import cast
import torch.nn as nn


class ExpertFusion(nn.Module):
    """
    Fuses embeddings from all three experts into unified market context.
    Strategy: Concatenation + MLP mixing
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        volatility_dim: int = 16,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        input_total = temporal_dim + spatial_dim + volatility_dim

        self.net = nn.Sequential(
            nn.Linear(input_total, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),  # Configurable dropout rate
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(
        self, emb_temporal: torch.Tensor, emb_spatial: torch.Tensor, emb_volatility: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns:
            fused: (batch, output_dim) global market context
        """
        # Concatenate along feature dimension (dim=1)
        concatenated = torch.cat([emb_temporal, emb_spatial, emb_volatility], dim=1)

        return cast(torch.Tensor, self.net(concatenated))
