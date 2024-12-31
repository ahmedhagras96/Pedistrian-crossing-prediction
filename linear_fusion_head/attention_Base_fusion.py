import torch
import torch.nn as nn
from base_fusion_head import BaseFusionHead

class AttentionFusionHead(BaseFusionHead):
    def __init__(self, feature_dim, num_heads=4, output_dim=1):
        """
        Attention-based fusion head.

        Args:
            feature_dim (int): Dimensionality of input features.
            num_heads (int): Number of attention heads.
            output_dim (int): Output dimensionality (default: 1 for binary classification).
        """
        super(AttentionFusionHead, self).__init__(feature_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        """
        Forward pass for attention-based fusion.

        Args:
            inputs: List of input tensors to process with attention.

        Returns:
            torch.Tensor: Output of the fusion head.
        """
        # Assuming inputs are already aligned; concatenate for attention
        concatenated = torch.cat([self.flatten(x) for x in inputs], dim=1).unsqueeze(0)
        attn_output, _ = self.attention(concatenated, concatenated, concatenated)
        x = self.relu(self.fc1(attn_output.squeeze(0)))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
