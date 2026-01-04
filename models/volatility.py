import torch
from typing import cast
import torch.nn as nn

from config.settings import Settings


class VolatilityExpert(nn.Module):
    """
    Volatility expert using autoencoder architecture.
    """

    def __init__(self, input_dim: int, settings: Settings, hidden_dim: int = 32):
        super().__init__()

        latent_dim = settings.hyperparams.latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            # No BatchNorm on latent usually, to allow distribution to form freely?
            # Or add it. Prompt says "Add BatchNorm between layers".
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, vol_metrics: torch.Tensor) -> torch.Tensor:
        """Returns only the latent embedding."""
        return self.encode(vol_metrics)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(z))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample MSE reconstruction error.
        
        Usage in Regime Detection:
        - Low error: The current volatility state is "normal" / seen during training.
        - High error: The current state is anomalous or unseen.
        
        This error score is used by the RegimeVeto to switch to safer/shadow modes
        when market conditions deviate significantly from the training distribution.
        """
        recon = self.reconstruct(x)
        error = torch.mean((x - recon) ** 2, dim=1)
        # CRITICAL-003: Harden against overflow/NaN
        error = torch.nan_to_num(error, nan=1e6, posinf=1e6, neginf=1e6)
        return error
