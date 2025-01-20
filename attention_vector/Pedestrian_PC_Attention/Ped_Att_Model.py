import torch.nn as nn


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(PointNetFeatureExtractor, self).__init__()
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

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Tensor of shape [B, N, 3] (Batch, Num Points, Features)
        Returns:
            torch.Tensor: Tensor of shape [B, output_dim] (Aggregated Features for each pedestrian)
        """
        B, N, _ = points.shape
        x = self.mlp1(points)  # Shape: [B, N, 256]
        x = x.transpose(1, 2)  # Shape: [B, 256, N]
        x = self.global_pool(x).squeeze(-1)  # Shape: [B, 256]
        x = self.fc(x)  # Shape: [B, output_dim]
        return x

