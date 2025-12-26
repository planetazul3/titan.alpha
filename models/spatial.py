import torch
from typing import cast
import torch.nn as nn

from config.settings import Settings
from models.blocks import Conv1DBlock
from models.attention import AdditiveAttention


class SpatialExpert(nn.Module):
    """
    Spatial expert for barrier/touch/run contracts.
    Architecture: Pyramidal 1D-CNN (kernels 3, 5, 15).
    """

    def __init__(self, settings: Settings, embedding_dim: int = 64):
        super().__init__()

        base_channels = settings.hyperparams.cnn_filters

        # Parallel blocks with different receptive fields could be concatenated
        # OR stacked. Prompt implies stacked "Pyramidal ... Block1 ... Block2 ... Block3"
        # with increasing kernels implies we are stacking?
        # But usually pyramidal implies multiscale parallel or U-Net.
        # "Block1: kernel=3 ... Block2: kernel=5... Block3: kernel=15"
        # If sequential: receptive field 3+4+14 = 21. If parallel: max is 15.
        # Prompt says "Analysis 'roughness' and geometry... Larger kernels capture momentum".
        # Let's assume sequential for depth features.

        self.block1 = Conv1DBlock(1, base_channels, kernel_size=3)
        self.block2 = Conv1DBlock(base_channels, base_channels * 2, kernel_size=5)
        self.block3 = Conv1DBlock(base_channels * 2, base_channels * 4, kernel_size=15)

        self.block3 = Conv1DBlock(base_channels * 2, base_channels * 4, kernel_size=15)

        # M07: Replace Global Average Pooling with Attention Pooling
        # This preserves temporal information and allows focusing on specific features
        self.attention = AdditiveAttention(hidden_dim=base_channels * 4)
        
        self.projector = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, ticks: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
           ticks: (batch, seq_len) -- wait, Conv1d needs (batch, channels, seq_len)
           If input is just 1D ticks, we assume (batch, seq_len) and unsqeeze channel.
           Prompt says input: "normalized tick series"
           And "ticks: (batch, 1, seq_len) normalized tick series" in prompt docstring.
           mask: Optional boolean mask (batch, seq_len) where 0/False is padding.
        """
        if ticks.dim() == 2:
            ticks = ticks.unsqueeze(1)  # (batch, 1, seq_len)

        # Ensure padding is zeroed out to prevent garbage from propagating via convolution
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, seq_len) matching ticks
            ticks = ticks * mask.unsqueeze(1)

        x = self.block1(ticks)
        x = self.block2(x)
        x = self.block3(x)

        # M07: Attention Pooling
        # Transpose to (batch, seq_len, channels) for attention
        x = x.permute(0, 2, 1)  
        
        # Apply attention pooling
        # context: (batch, channels)
        # Note: CNN reduces sequence length only if stride > 1. 
        # Here strides are default 1 (presumably in Conv1DBlock), so seq_len is preserved.
        # However, if padding/kernel size changes length, mask alignment is tricky.
        # Assuming "same" padding or simple causal conv where L_out = L_in.
        context, _ = self.attention(x, mask=mask)

        embedding = self.projector(context)
        return cast(torch.Tensor, embedding)
