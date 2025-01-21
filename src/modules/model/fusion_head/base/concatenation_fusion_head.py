import torch
import torch.nn as nn

from modules.model.fusion_head.base.base_fusion_head import BaseFusionHead


class ConcatenationFusionHead(BaseFusionHead):
    """
    Concatenation-based fusion head for combining multiple input tensors
    by flattening and concatenating them along the batch dimension.
    """

    def __init__(self, feature_dim: int, output_dim: int = 1):
        """
        Initialize the ConcatenationFusionHead.

        Args:
            feature_dim (int): Dimensionality of each input feature.
            output_dim (int): Dimensionality of the output layer. Defaults to 1 
                (e.g., for binary classification).
        """
        super().__init__(feature_dim)

        # Assuming 3 input tensors are concatenated, hence the multiplier '3'
        self.fc1 = nn.Linear(feature_dim * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for concatenation fusion.

        Args:
            *inputs (torch.Tensor): Variable number of input tensors to concatenate.
                Each input is expected to have the same feature_dim.

        Returns:
            torch.Tensor: The output tensor after the concatenation and final activation.
        """
        concatenated = self._concatenate_and_flatten_inputs(*inputs)
        x = self.relu(self.fc1(concatenated))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

    def _concatenate_and_flatten_inputs(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Private helper method to flatten and concatenate input tensors.

        Args:
            *inputs (torch.Tensor): Input tensors to be flattened and concatenated.

        Returns:
            torch.Tensor: A single concatenated tensor of shape (batch_size, feature_dim * num_inputs).
        """
        return torch.cat([self.flatten(x) for x in inputs], dim=1)
