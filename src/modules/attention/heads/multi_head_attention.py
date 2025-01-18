import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.config.logger import LoggerUtils


class MultiHeadAttention(nn.Module):
    """
    A PyTorch implementation of the Multi-Head Attention mechanism.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention heads.
        query (nn.Linear): Linear layer to compute query vectors.
        key (nn.Linear): Linear layer to compute key vectors.
        value (nn.Linear): Linear layer to compute value vectors.
        out (nn.Linear): Linear layer for combining attention outputs.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, input_dim, num_heads, dropout=0.1):
        """
        Initialize the MultiHeadAttention module.

        Args:
            input_dim (int): Dimensionality of the input feature vectors.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention weights.
        """
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"

        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing MultiHeadAttention with input_dim={input_dim}, num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Define layers for query, key, and value projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output projection and dropout
        self.out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        self.logger.info("MultiHeadAttention initialized successfully")

    def forward(self, x):
        """
        Perform the forward pass of the Multi-Head Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_pedestrians, input_dim].

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Attention output tensor of shape [batch_size, num_pedestrians, input_dim].
                - attention_weights (torch.Tensor): Attention weights of shape
                                                     [batch_size, num_heads, num_pedestrians, num_pedestrians].
        """
        try:
            batch_size, num_pedestrians, input_dim = x.shape
            self.logger.debug(f"Forward pass with input shape: {x.shape}")

            # Compute query, key, and value matrices
            Q = self.query(x).view(batch_size, num_pedestrians, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, num_pedestrians, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, num_pedestrians, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            attention_weights = F.softmax(scores, dim=-1)
            self.logger.debug(f"Attention weights computed with shape: {attention_weights.shape}")

            # Apply dropout to attention weights
            weighted_values = torch.matmul(self.dropout(attention_weights), V)

            # Concatenate heads and apply output projection
            weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, num_pedestrians, input_dim)
            output = self.out(weighted_values)

            self.logger.debug(f"Output shape: {output.shape}")
            return output, attention_weights

        except Exception as e:
            self.logger.error(f"Error during forward pass: {e}")
            raise e
