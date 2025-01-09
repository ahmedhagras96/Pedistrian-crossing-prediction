import torch.nn as nn

from modules.attention_vector.point_cloud_attention.centroid_aware_voxelization import CentroidAwareVoxelization
from modules.attention_vector.point_cloud_attention.lightweight_self_attention import LightweightSelfAttentionLayer


class PointCloudAttentionModel(nn.Module):
    """
    A model for processing point clouds using centroid-aware voxelization and lightweight self-attention.

    Args:
        embed_dim (int): Dimensionality of the feature embeddings.
        kernel_size (int): Size of the kernel for local self-attention.
        stride (int): Stride for kernel mapping in the attention layer.
        dilation (int): Dilation factor for kernel mapping in the attention layer.
        num_heads (int): Number of attention heads in the attention layer.

    Attributes:
        voxelization (CentroidAwareVoxelization): Module for voxelizing point cloud data and generating features.
        attention_layer (LightweightSelfAttentionLayer): Attention mechanism for aggregating voxel features.
    """

    def __init__(self, embed_dim, kernel_size=3, stride=1, dilation=1, num_heads=4):
        super(PointCloudAttentionModel, self).__init__()

        # Centroid-aware voxelization module
        self.voxelization = CentroidAwareVoxelization(embed_dim=embed_dim)

        # Lightweight self-attention layer
        self.attention_layer = LightweightSelfAttentionLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, point_cloud):
        """
        Forward pass of the PointCloudAttentionModel.

        Args:
            point_cloud (torch.Tensor): Input point cloud of shape [B, N, 3], where B is the batch size,
                                        N is the number of points, and 3 represents (x, y, z) coordinates.

        Returns:
            tuple:
                - output_features (torch.Tensor): Processed features of shape [B, embed_dim].
                - attention_weights (torch.Tensor): Attention weights of shape [B, M, num_heads],
                                                     where M is the number of unique voxels.
        """
        # Apply centroid-aware voxelization
        (
            aggregated_features,  # [B, max_voxels, embed_dim]
            norm_points,  # [B, max_voxels, 3]
            voxel_centroids,  # [num_voxels, 3]
            voxel_counts,  # [num_voxels]
            positional_embeddings,  # [B * N, embed_dim]
            batch_indices,  # [B, N]
        ) = self.voxelization(point_cloud)

        # Apply lightweight self-attention
        output_features, attention_weights = self.attention_layer(
            aggregated_features, norm_points, batch_indices
        )

        return output_features, attention_weights
