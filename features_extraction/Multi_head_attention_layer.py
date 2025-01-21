import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, output_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size (num_pedestrians), input_dim)
        batch_size, input_dim = x.shape

        # Compute Q, K, V
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        weighted_values = torch.matmul(self.dropout(attention_weights), V)

        # Combine heads and project to output dimension
        weighted_values = weighted_values.view(batch_size, input_dim)
        output = self.out(weighted_values)

        # Output shape: (batch_size (num_pedestrians), output_dim)
        return output, attention_weights