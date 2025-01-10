import torch
import torch.nn as nn

from modules.utilities.logger import LoggerUtils


class PointNetFeatureExtractor(nn.Module):
    """
    A neural network model for extracting features from point cloud data using PointNet.
    """

    def __init__(self, input_dim=3, output_dim=64):
        """
        Initialize the PointNet model.

        Args:
            input_dim (int): Dimensionality of the input features (e.g., 3 for (x, y, z) coordinates).
            output_dim (int): Dimensionality of the output feature embeddings.
        """
        super(PointNetFeatureExtractor, self).__init__()
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

        self.logger.info(f"Initializing PointNet with input_dim={input_dim}, output_dim={output_dim}")

        # MLP for local feature extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # Global max pooling for aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers for global feature extraction
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, points):
        """
        Compute features for input point cloud data.

        Args:
            points (torch.Tensor): Tensor of shape [B, N, input_dim], where B is the batch size,
                                   N is the number of points, and input_dim is the dimensionality of the points.

        Returns:
            torch.Tensor: Tensor of shape [B, output_dim] (aggregated features for each batch).
        """
        B, N, _ = points.shape
        self.logger.debug(f"Forward pass with input shape: {points.shape}")

        # Apply MLP for local feature extraction
        x = self.mlp1(points)  # Shape: [B, N, 256]
        x = x.transpose(1, 2)  # Shape: [B, 256, N]

        # Global max pooling for feature aggregation
        x = self.global_pool(x).squeeze(-1)  # Shape: [B, 256]

        # Fully connected layers for global feature extraction
        x = self.fc(x)  # Shape: [B, output_dim]
        self.logger.debug(f"Feature extraction complete with output shape: {x.shape}")
        return x

    def extract_features(self, points_batch):
        """
        Extract features for a batch of point clouds.

        Args:
            points_batch (torch.Tensor): Input batch of point clouds.

        Returns:
            torch.Tensor: Extracted features for the batch.
        """
        self.logger.info(f"Extracting features for batch of size {points_batch.size(0)}")
        return self.forward(points_batch)
