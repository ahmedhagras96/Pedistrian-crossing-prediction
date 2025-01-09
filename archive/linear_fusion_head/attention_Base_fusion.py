import torch
import torch.nn as nn
from base_fusion_head import BaseFusionHead

class AttentionFusionHead(BaseFusionHead):
    def __init__(self, vector_dim=64, num_heads=4):
        """
        End-to-end neural network with attention-based fusion.

        Args:
            vector_dim (int): Dimensionality of attention vectors from each source (default: 64).
            num_heads (int): Number of attention heads for fusion.
        """
        super(AttentionFusionHead, self).__init__()

        # Attention-based fusion layer
        self.fusion_attention = nn.MultiheadAttention(embed_dim=vector_dim, num_heads=num_heads)

        # Fully connected layers after fusion
        self.fc1 = nn.Linear(vector_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, source1, source2, source3):
        """
        Forward pass of the network.

        Args:
            source1 (torch.Tensor): Attention vectors from 3D constructed data (batch_size, seq_len, vector_dim).
            source2 (torch.Tensor): Attention vectors from 3D pedestrian point cloud (batch_size, seq_len, vector_dim).
            source3 (torch.Tensor): Attention vectors from pedestrian features (batch_size, seq_len, vector_dim).

        Returns:
            torch.Tensor: Binary classification output (batch_size, 1).
        """
        # Concatenate inputs along the sequence dimension
        concatenated_inputs = torch.cat([source1, source2, source3], dim=1)  # (batch_size, 3 * seq_len, vector_dim)

        # Apply attention-based fusion
        fusion_output, _ = self.fusion_attention(concatenated_inputs, concatenated_inputs, concatenated_inputs)  # (batch_size, 3 * seq_len, vector_dim)

        # Average pooling across the sequence length
        pooled_output = torch.mean(fusion_output, dim=1)  # (batch_size, vector_dim)

        # Fully connected layers
        x = self.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.sigmoid(self.fc3(x))  # (batch_size, 1)

        return output
