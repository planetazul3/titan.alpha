"""
Reusable neural network building blocks.

This module provides common building blocks used throughout the model architecture:
- BiLSTMBlock: Bidirectional LSTM with proper initialization
- Conv1DBlock: 1D convolution with batch norm and pooling
- MLPBlock: Flexible multi-layer perceptron

All blocks include proper weight initialization and configurable activations.

Example:
    >>> from models.blocks import BiLSTMBlock, MLPBlock
    >>> lstm = BiLSTMBlock(input_size=64, hidden_size=128)
    >>> mlp = MLPBlock([256, 128, 64], dropout=0.1)
"""

import logging
from typing import Any, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM wrapper with proper initialization.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden size (output is 2*hidden_size due to bidirectionality)
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate between layers (ignored if num_layers==1)

    Input shape: (batch, seq_len, input_size)
    Output shape: (batch, seq_len, 2*hidden_size)
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError(
                f"Sizes must be positive: input_size={input_size}, hidden_size={hidden_size}"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights with Xavier/Orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BiLSTM.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size)
            lengths: Optional sequence lengths for packing (not implemented)

        Returns:
            output: LSTM output, shape (batch, seq_len, 2*hidden_size)
            (h_n, c_n): Final hidden and cell states
        """
        if lengths is not None:
            # Sequence packing not implemented - variable-length sequences should be handled
            # at the dataloader level via padding/masking for batch processing efficiency.
            # This keeps the model architecture simple and delegates sequence handling
            # to the data pipeline where it belongs.
            logger.warning(
                "Sequence packing not implemented. Use dataloader padding/masking for "
                "variable-length sequences. Ignoring lengths parameter."
            )
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class Conv1DBlock(nn.Module):
    """
    Single convolutional block with batch norm and pooling.

    Architecture: Conv1d -> BatchNorm1d -> Activation -> Pool

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding mode ('same' or int)
        activation: Activation function ('silu', 'relu', or None)
        pool_size: Max pooling size (None to disable)

    Input shape: (batch, in_channels, seq_len)
    Output shape: (batch, out_channels, seq_len // pool_size)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        activation: str = "silu",
        pool_size: int | None = None,
    ):
        super().__init__()
        # 'same' padding requires PyTorch >= 1.9
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act: nn.Module
        
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        self.pool = nn.MaxPool1d(pool_size) if pool_size else nn.Identity()

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class MLPBlock(nn.Module):
    """
    Flexible Multi-Layer Perceptron with configurable depth and regularization.

    Args:
        layer_sizes: List of layer dimensions [input_dim, hidden1, ..., output_dim]
        activation: Activation function ('silu' or 'relu')
        dropout: Dropout rate (0 to disable)
        use_layer_norm: Whether to use layer normalization

    Input shape: (*, layer_sizes[0])
    Output shape: (*, layer_sizes[-1])
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "silu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError(f"layer_sizes must have at least 2 elements, got {len(layer_sizes)}")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError(f"All layer sizes must be positive: {layer_sizes}")

        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # No activation/dropout/norm on last layer
            if i < len(layer_sizes) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(layer_sizes[i + 1]))

                if activation == "silu":
                    layers.append(nn.SiLU())
                elif activation == "relu":
                    layers.append(nn.ReLU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights with Kaiming initialization."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return cast(torch.Tensor, self.net(x))
