import torch.nn as nn
import torch

from attention_vector.point_cloud_attn_vector.modules import LightweightSelfAttentionLayer
from attention_vector.point_cloud_attn_vector.modules import CentroidAwareVoxelization

class PointCloudAttentionModel(nn.Module):
    """
    A model that applies centroid-aware voxelization to point cloud data,
    then processes the voxelized features with a lightweight self-attention
    mechanism optimized for sparse tensors.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 3, num_heads: int = 4, max_voxel_grid_size: int = int(1e5), sparse_ratio: float = 0.5):
        """
        Initialize the point cloud attention model.

        Args:
            embed_dim (int):
                Dimensionality of the embedding space used in centroid-aware
                voxelization and self-attention.
            kernel_size (int, optional):
                Neighborhood kernel size for the self-attention layer. Defaults to 3.
            num_heads (int, optional):
                Number of attention heads. Defaults to 4.
        """
        super().__init__()

        # Initialize Centroid-Aware Voxelization Module
        self.centroid_aware_voxelization = CentroidAwareVoxelization(embed_dim=embed_dim, max_voxel_grid_size=max_voxel_grid_size)

        # Initialize Lightweight Self-Attention Layer
        self.attn_layer = LightweightSelfAttentionLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            sparse_ratio=sparse_ratio
        )


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        The input is first voxelized and aggregated using a centroid-aware approach.
        The resulting sparse features are then passed through a lightweight self-attention layer.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, N, 3), where B is the batch size and
                N is the number of points per batch.

        Returns:
            out (torch.Tensor):
                Aggregated output features of shape [B, embed_dim] after
                max pooling in the self-attention layer.
            attn_weights (torch.Tensor):
                Attention map of shape [B, M, num_heads], where M is the number of
                unique voxel coordinates in the batch.
        """
        # Perform centroid-aware voxelization
        (sparse_features,
         norm_points) = self.centroid_aware_voxelization(x)

        # Process voxelized sparse features through attention layer
        out, attn_weights = self.attn_layer(sparse_features, norm_points)

        return out, attn_weights
