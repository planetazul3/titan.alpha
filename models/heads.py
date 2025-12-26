import torch
from typing import cast
import torch.nn as nn

from config.settings import Settings


class ContractHead(nn.Module):
    """
    Generic output head for a single contract type.
    Returns raw logit (no Sigmoid) for numerical stability with BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []

        if hidden_dim:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logit: (batch, 1) raw logit for this contract
        """
        return cast(torch.Tensor, self.net(context))


def create_contract_heads(input_dim: int, settings: Settings) -> nn.ModuleDict:
    """
    Factory function to create all contract heads.
    """
    dropout = settings.hyperparams.head_dropout

    # We can use slightly different architectures per head if needed
    heads = nn.ModuleDict(
        {
            "rise_fall": ContractHead(input_dim, hidden_dim=128, dropout=dropout),
            "touch": ContractHead(input_dim, hidden_dim=64, dropout=dropout),
            "range": ContractHead(input_dim, hidden_dim=64, dropout=dropout),
        }
    )

    return heads
