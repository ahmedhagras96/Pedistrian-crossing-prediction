import torch
import torch.nn as nn

from modules.config.logger import Logger
from src.modules.attention.pointcloud_attention.layers.centroid_aware_voxelization import CentroidAwareVoxelization
from src.modules.attention.pointcloud_attention.layers.light_weight_self_attention import LightweightSelfAttentionLayer


class PointCloudAttentionModel(nn.Module):
    """
    A model that applies centroid-aware voxelization to point cloud data,
    then processes the voxelized features with a lightweight self-attention
    mechanism.
    """

    def __init__(self, embed_dim: int, kernel_size: int = 3, num_heads: int = 4):
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

        # Create a logger for this class
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"embed_dim={embed_dim}, kernel_size={kernel_size}, "
                         f"num_heads={num_heads}")

        self.centroid_aware_voxelization = CentroidAwareVoxelization(embed_dim=embed_dim)
        self.attn_layer = LightweightSelfAttentionLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size
        )

        # self.logger.debug(f"{self.__class__.__name__} successfully initialized.")

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        The input is first voxelized and aggregated using a centroid-aware approach.
        The resulting features are then passed through a lightweight self-attention layer.

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
        # self.logger.debug(f"Forward called with input x of shape {x.shape}")

        (aggregated_features,
         norm_points,
         voxel_centroids,
         voxel_counts,
         pos_embs,
         batch_ids) = self.centroid_aware_voxelization(x)

        # self.logger.debug("Features voxelized and aggregated. "
        #                   f"aggregated_features shape: {aggregated_features.shape}, "
        #                   f"norm_points shape: {norm_points.shape}, "
        #                   f"batch_ids shape: {batch_ids.shape}")

        # Pass voxelized features through the attention layer
        out, attn_weights = self.attn_layer(aggregated_features, norm_points, batch_ids)

        # self.logger.debug("Attention layer processed features. "
        #                   f"Output shape: {out.shape}, Attention map shape: {attn_weights.shape}")

        return out, attn_weights
