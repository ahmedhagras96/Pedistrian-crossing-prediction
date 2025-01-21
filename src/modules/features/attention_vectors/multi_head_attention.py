from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer as described in "Attention is All You Need".

    Args:
        input_dim (int): Dimensionality of the input features.
        num_heads (int): Number of attention heads.
        output_dim (int): Dimensionality of the output features.
        dropout (float, optional): Dropout probability. Default: 0.1.
    """

    def __init__(self, input_dim: int, num_heads: int, output_dim: int, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output tensor of shape (batch_size, output_dim).
                - Attention weights tensor of shape 
                  (batch_size, num_heads, head_dim, head_dim).
        """
        batch_size, input_dim = x.shape

        # Compute Q, K, V projections for each head
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention computation
        scaling_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor
        attention_weights = F.softmax(scores, dim=-1)
        weighted_values = torch.matmul(self.dropout(attention_weights), V)

        # Combine heads and project to output dimension
        combined = weighted_values.view(batch_size, input_dim)
        output = self.out(combined)

        return output, attention_weights
