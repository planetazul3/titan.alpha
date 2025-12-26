"""
Temporal Fusion Transformer (TFT) Components.

Implements the Temporal Fusion Transformer architecture for time series
forecasting in trading applications.

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon 
Time Series Forecasting" (Lim et al., 2021)
https://arxiv.org/abs/1912.09363

Key components:
- GatedResidualNetwork (GRN): Skip connections with gating for gradient flow
- VariableSelectionNetwork (VSN): Learned feature importance weights
- InterpretableMultiHeadAttention: Attention with visualizable weights
- TemporalFusionTransformer: Full TFT encoder module
- MultiHorizonHead: Multi-step ahead forecasting output

ARCHITECTURAL PRINCIPLE:
TFT provides superior long-range temporal modeling compared to LSTM
while maintaining interpretability through attention weights and
feature importance scores.

Example:
    >>> from models.tft import TemporalFusionTransformer
    >>> tft = TemporalFusionTransformer(input_size=10, hidden_size=64)
    >>> output, attention_weights = tft(candle_features)
"""

import logging
import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiHorizonHead(nn.Module):
    """
    Multi-horizon prediction head for TFT.
    
    Produces predictions at multiple future horizons (e.g., 1-step, 5-step, 10-step).
    Supports quantile outputs for uncertainty estimation.
    
    Example:
        >>> head = MultiHorizonHead(hidden_size=64, horizons=[1, 5, 10])
        >>> predictions = head(tft_output)  # {1: pred_1, 5: pred_5, 10: pred_10}
    """
    
    def __init__(
        self,
        hidden_size: int,
        horizons: list[int] = [1, 5, 10],
        output_size: int = 1,
        quantiles: list[float] | None = None,
    ):
        """
        Initialize multi-horizon head.
        
        Args:
            hidden_size: Input hidden dimension from TFT
            horizons: List of forecast horizons (e.g., [1, 5, 10])
            output_size: Output dimension per horizon
            quantiles: Optional quantiles for probabilistic forecasting
        """
        super().__init__()
        self.horizons = horizons
        self.output_size = output_size
        self.quantiles = quantiles or [0.5]  # Default to median
        
        n_quantiles = len(self.quantiles)
        
        # Separate head for each horizon
        self.horizon_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size * n_quantiles),
            )
            for h in horizons
        })
        
        logger.info(f"MultiHorizonHead: horizons={horizons}, quantiles={self.quantiles}")
    
    def forward(
        self, 
        hidden: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """
        Produce multi-horizon predictions.
        
        Args:
            hidden: TFT output [batch, hidden_size]
        
        Returns:
            Dictionary mapping horizon -> predictions [batch, output_size, n_quantiles]
        """
        predictions = {}
        
        for h in self.horizons:
            pred = self.horizon_heads[str(h)](hidden)
            
            # Reshape for quantiles, preserving all leading dimensions
            # pred is [..., output_size * n_quantiles]
            # We want [..., output_size, n_quantiles]
            
            new_shape = list(pred.shape[:-1]) + [self.output_size, len(self.quantiles)]
            pred = pred.view(*new_shape)
            
            predictions[h] = pred
        
        return predictions
    
    def get_point_predictions(
        self, 
        hidden: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Get point predictions (median quantile)."""
        all_preds = self.forward(hidden)
        
        # Find median quantile index
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else 0
        
        return {h: pred[:, :, median_idx] for h, pred in all_preds.items()}


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) for controlled information flow.
    
    GLU(x) = σ(Wx + b) ⊙ (Vx + c)
    
    The sigmoid gate controls how much information passes through.
    """
    
    def __init__(self, input_size: int, output_size: int | None = None):
        """
        Initialize GLU.
        
        Args:
            input_size: Input dimension
            output_size: Output dimension (defaults to input_size)
        """
        super().__init__()
        output_size = output_size or input_size
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU."""
        x = self.fc(x)
        return x[:, :, :self.output_size] * torch.sigmoid(x[:, :, self.output_size:])


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for non-linear processing with skip connections.
    
    Architecture:
    1. Linear projection (optional, if input != output size)
    2. ELU activation
    3. Linear
    4. Dropout
    5. Gated Linear Unit
    6. Layer Normalization
    7. Skip connection
    
    Optionally includes context vector for static enrichment.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize GRN.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (defaults to hidden_size)
            context_size: Context vector dimension (optional)
            dropout: Dropout rate
        """
        super().__init__()
        output_size = output_size or hidden_size
        
        # Main path
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Context enrichment (optional)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        
        # Skip connection projection (if input != output)
        self.skip_layer = nn.Linear(input_size, output_size) if input_size != output_size else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply GRN.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            context: Optional context vector [batch, context_size]
        
        Returns:
            Output tensor [batch, seq_len, output_size]
        """
        # Skip connection
        skip = self.skip_layer(x) if self.skip_layer else x
        
        # Main path
        hidden = F.elu(self.fc1(x))
        
        # Add context if provided
        if context is not None and self.context_fc is not None:
            # Expand context to match sequence length
            if context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            hidden = hidden + self.context_fc(context)
        
        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)
        
        # Residual connection with layer norm
        return cast(torch.Tensor, self.layer_norm(hidden + skip))


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for learning feature importance.
    
    Uses softmax attention over input features to learn which
    features are most relevant for the prediction task.
    
    This enables interpretability: we can examine feature weights
    to understand what the model considers important.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize VSN.
        
        Args:
            num_features: Number of input features to select from
            hidden_size: Hidden dimension for processing
            context_size: Optional context vector dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # GRN for each input feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_features)
        ])
        
        # Selection GRN (produces softmax weights)
        self.selection_grn = GatedResidualNetwork(
            num_features * hidden_size,
            hidden_size,
            num_features,
            context_size=context_size,
            dropout=dropout,
        )
        
        # Output projection
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply variable selection.
        
        Args:
            x: Input features [batch, seq_len, num_features]
            context: Optional context vector
        
        Returns:
            Tuple of:
            - Selected features [batch, seq_len, hidden_size]
            - Feature weights [batch, seq_len, num_features]
        """
        batch_size, seq_len, _ = x.shape
        
        # Process each feature separately
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            feature = x[:, :, i:i+1]  # [batch, seq_len, 1]
            processed = grn(feature)   # [batch, seq_len, hidden_size]
            processed_features.append(processed)
        
        # Stack and flatten for selection
        stacked = torch.stack(processed_features, dim=2)  # [batch, seq_len, num_features, hidden_size]
        flattened = stacked.reshape(batch_size, seq_len, -1)  # [batch, seq_len, num_features * hidden_size]
        
        # Compute selection weights
        selection_weights = self.selection_grn(flattened, context)  # [batch, seq_len, num_features]
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Apply weights
        weighted = stacked * selection_weights.unsqueeze(-1)  # [batch, seq_len, num_features, hidden_size]
        combined = weighted.sum(dim=2)  # [batch, seq_len, hidden_size]
        
        # Final processing
        output = self.output_grn(combined)
        
        return output, selection_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for temporal patterns.
    
    Unlike standard multi-head attention which concatenates heads,
    this version additively combines heads to produce interpretable
    attention weights that can be visualized.
    
    Each head attends to different temporal patterns, and the final
    attention weights represent aggregate temporal importance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize interpretable attention.
        
        Args:
            hidden_size: Input/output dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply interpretable multi-head attention.
        
        Args:
            query: Query tensor [batch, seq_len, hidden_size]
            key: Key tensor (defaults to query for self-attention)
            value: Value tensor (defaults to query for self-attention)
            mask: Optional attention mask
        
        Returns:
            Tuple of:
            - Output tensor [batch, seq_len, hidden_size]
            - Attention weights [batch, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.output(context)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        return output, avg_attention


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for time series prediction.
    
    Full TFT architecture:
    1. Variable Selection Network for feature importance
    2. LSTM encoder for local temporal processing
    3. Gated skip connections
    4. Interpretable multi-head attention for long-range patterns
    5. Position-wise feed-forward network
    
    Outputs:
    - Hidden state representation for downstream heads
    - Attention weights for interpretability
    - Feature importance weights for interpretability
    
    Example:
        >>> tft = TemporalFusionTransformer(input_size=10, hidden_size=64)
        >>> output, attention_weights, feature_weights = tft(candles)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize TFT.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension throughout
            num_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            num_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        
        # LSTM encoder for local processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )
        
        # Gate for LSTM output
        self.lstm_gate = GatedLinearUnit(hidden_size * 2, hidden_size)
        
        # GRN after LSTM
        self.post_lstm_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        
        # Interpretable Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        
        # Final output GRN
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"TFT initialized: input={input_size}, hidden={hidden_size}, heads={num_heads}")
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply TFT encoder.
        
        Args:
            x: Input features [batch, seq_len, input_size]
            mask: Optional attention mask [batch, seq_len]
        
        Returns:
            Tuple of:
            - Output embedding [batch, hidden_size]
            - Attention weights [batch, seq_len, seq_len]
            - Feature importance weights [batch, seq_len, num_features]
        """
        # Variable selection
        selected, feature_weights = self.vsn(x)  # [batch, seq_len, hidden_size]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(selected)  # [batch, seq_len, hidden_size*2]
        lstm_out = self.lstm_gate(lstm_out)  # [batch, seq_len, hidden_size]
        
        # Skip connection from VSN
        lstm_out = self.post_lstm_grn(lstm_out) + selected
        
        # Self-attention for long-range patterns
        # H03: Pass mask to attention
        # Mask needs to be broadcastable to [batch, heads, seq, seq] or handled by attention
        attn_mask = None
        if mask is not None:
             # Mask keys (columns) where mask == 0
             # Shape: [batch, 1, 1, seq_len]
             attn_mask = mask.unsqueeze(1).unsqueeze(1)

        attention_out, attention_weights = self.attention(lstm_out, mask=attn_mask)
        
        # Skip connection
        attention_out = self.post_attention_grn(attention_out) + lstm_out
        
        # Final processing
        output = self.output_grn(attention_out)
        output = self.layer_norm(output)
        
        # Return full sequence for multi-horizon forecasting
        # Do NOT pool (mean) here - that destroys temporal dynamics!
        # [batch, seq_len, hidden_size]
        return output, attention_weights, feature_weights
    
    def get_feature_importance(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get feature importance weights for interpretability.
        
        Args:
            x: Input features
        
        Returns:
            Feature importance [batch, num_features]
        """
        _, _, feature_weights = self.forward(x)
        # Average across sequence for global importance
        return feature_weights.mean(dim=1)
