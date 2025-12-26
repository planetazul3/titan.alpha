import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention for sequential data.
    Input: (batch, seq_len, hidden_dim)
    Output: (context_vector, attention_weights)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(
        self, encoder_outputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: (B, T, H)
        # We can implement simple self-attention where query is a learned parameter or
        # based on usage, usually we attend to all steps.
        # For simple "summarize sequence" attention (often used in RNNs):
        # score = V * tanh(W * h)

        # W(h): (B, T, H)
        attn_score = self.V(torch.tanh(self.W(encoder_outputs)))  # (B, T, 1)

        if mask is not None:
            # mask: (B, T)
            attn_score = attn_score.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        attn_weights = F.softmax(attn_score, dim=1)  # (B, T, 1)

        # context: sum(weights * h) -> (B, H)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)

        return context, attn_weights.squeeze(-1)


class ScaledDotProductAttention(nn.Module):
    """
    Transformer-style scaled dot-product attention.
    Formula: softmax(QK^T / sqrt(d_k)) @ V
    """

    def __init__(self, temperature: float | None = None):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k, v: (B, T, H) or (B, H) depending on usage
        # Assuming (B, T, H)

        d_k = k.size(-1)
        temp = self.temperature if self.temperature else d_k**0.5

        # scores: (B, T_q, T_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / temp

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
