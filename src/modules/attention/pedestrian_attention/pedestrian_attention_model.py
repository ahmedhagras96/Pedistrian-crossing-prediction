import torch
import torch.nn as nn

from modules.config.logger import Logger


class PedestrianAttentionPointNetModel(nn.Module):
    """
    A PointNet-like architecture for extracting features from 3D pedestrian points.
    Applies an MLP to each point, performs global pooling to aggregate features,
    and then uses a final MLP to produce the output representation.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 64):
        """
        Initialize the model components.

        Args:
            input_dim (int, optional): Dimensionality of the input point features.
                Defaults to 3, typically (x, y, z).
            output_dim (int, optional): Dimensionality of the final output.
                Defaults to 64.
        """
        super(PedestrianAttentionPointNetModel, self).__init__()

        # Initialize logger
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"input_dim={input_dim}, output_dim={output_dim}")

        # Feature extraction MLP: [input_dim -> 64 -> 128 -> 256]
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Output MLP: [256 -> 128 -> output_dim]
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PointNet-based feature extractor.
    
        Args:
            points (torch.Tensor): 
                A tensor of shape (B, N, 3) where
                B is the batch size, 
                N is the number of points, 
                3 is the dimensionality of each point.
    
        Returns:
            torch.Tensor: 
                A tensor of shape (B, output_dim) which represents
                the aggregated features for each batch element.
        """
        # Log shapes (useful for debugging)
        B, N, _ = points.shape
        self.logger.debug(f"Forward called with input of shape: {points.shape}")

        # Extract point-wise features: shape [B, N, 256]
        x = self.feature_extractor(points)

        # Transpose to [B, 256, N] for 1D global pooling
        x = x.transpose(1, 2)

        # Global max pooling: shape [B, 256]
        x = self.global_pool(x).squeeze(-1)

        # Final output layer: shape [B, output_dim]
        x = self.output_layer(x)

        return x
