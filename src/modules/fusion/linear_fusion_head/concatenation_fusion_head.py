import torch
import torch.nn as nn

from base_fusion_head import BaseFusionHead


class ConcatenationFusionHead(BaseFusionHead):
    def __init__(self, feature_dim, output_dim=1):
        """
        Concatenation-based fusion head.

        Args:
            feature_dim (int): Dimensionality of input features.
            output_dim (int): Output dimensionality (default: 1 for binary classification).
        """
        super(ConcatenationFusionHead, self).__init__(feature_dim)
        self.fc1 = nn.Linear(feature_dim * 3, 128)  # Assuming 3 input features concatenated
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        """
        Forward pass for concatenation fusion.

        Args:
            inputs: List of input tensors to concatenate.

        Returns:
            torch.Tensor: Output of the fusion head.
        """
        # Flatten and concatenate inputs
        concatenated = torch.cat([self.flatten(x) for x in inputs], dim=1)
        x = self.relu(self.fc1(concatenated))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
