import torch
import torch.nn as nn

from modules.config.logger import Logger


class PedestrianPointNetFeatureExtractor(nn.Module):
    """
    A PointNet-based feature extractor for 3D point cloud data.
    """

    _logger = Logger.get_logger("PedestrianPointNetFeatureExtractor")

    def __init__(self, input_dim: int = 3, output_dim: int = 64):
        """
        Initialize the PointNetFeatureExtractor.

        Args:
            input_dim (int): The dimensionality of the input features (e.g., 3 for x,y,z).
            output_dim (int): The dimensionality of the final output features.
        """
        super(PedestrianPointNetFeatureExtractor, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self._logger.info(
            f"Initialized PedestrianPointNetFeatureExtractor with input_dim={input_dim}, output_dim={output_dim}"
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PointNet feature extractor.

        Args:
            points (torch.Tensor): 
                A tensor of shape [B, N, input_dim] where B is the batch size, 
                N is the number of points, and input_dim is the point dimension.

        Returns:
            torch.Tensor: 
                A tensor of shape [B, output_dim], representing the extracted features 
                for each sample in the batch.
        """
        self._logger.debug(f"Forward called with points shape: {points.shape}")
        B, N, _ = points.shape

        # Pass through MLP
        x = self.mlp1(points)  # Shape: [B, N, 256]
        # Prepare for global pooling
        x = x.transpose(1, 2)   # Shape: [B, 256, N]
        x = self.global_pool(x).squeeze(-1)  # Shape: [B, 256]
        # Final fully-connected layer
        x = self.fc(x)  # Shape: [B, output_dim]

        self._logger.debug(f"Feature shape after forward pass: {x.shape}")
        return x
