import numpy as np
import open3d as o3d
import torch
import torch.nn as nn


class CentroidAwareVoxelization(nn.Module):
    """
    A module for voxelizing point cloud data with centroid-aware positional embeddings and feature aggregation.

    Args:
        embed_dim (int): Dimensionality of the embedding space for positional and voxel features.

    Attributes:
        embed_dim (int): Dimensionality of the embedding space.
        pos_enc_mlp (nn.Sequential): MLP for generating positional embeddings from normalized point coordinates.
        feature_mlp (nn.Sequential): MLP for generating voxel features from input features and positional embeddings.
    """
    def __init__(self, embed_dim):
        super(CentroidAwareVoxelization, self).__init__()
        self.embed_dim = embed_dim

        # MLP for positional embeddings
        self.pos_enc_mlp = nn.Sequential(
            nn.Linear(3, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # MLP for voxel feature generation
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, voxel_size: float = 0.05) -> tuple[
        torch.Tensor,  # padded_aggregated_features
        torch.Tensor,  # padded_norm_points
        torch.Tensor,  # voxel_centroids
        torch.Tensor,  # voxel_counts
        torch.Tensor,  # positional_embeddings
        torch.Tensor  # batch_indices
    ]:

        """
            Compute voxelized representation and features from point cloud data.
    
            Args:
                points (torch.Tensor): Tensor of shape (B, N, 3) representing point coordinates for B batches.
                voxel_size (float): Size of the voxel grid.
    
            Returns:
                tuple: A tuple containing:
                    - padded_aggregated_features (torch.Tensor): Padded voxel features of shape (B, max_voxels, embed_dim).
                    - padded_norm_points (torch.Tensor): Padded normalized points of shape (B, max_voxels, 3).
                    - voxel_centroids (torch.Tensor): Centroids of voxels of shape (num_voxels, 3).
                    - voxel_counts (torch.Tensor): Number of points in each voxel.
                    - positional_embeddings (torch.Tensor): Positional embeddings of shape (B * N, embed_dim).
                    - batch_indices (torch.Tensor): Batch indices for each point of shape (B, N).
            """
        batch_size, num_points, _ = points.shape
        device = points.device

        # Flatten batch dimension
        flat_points = points.view(-1, 3)  # Shape: (B * N, 3)
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)  # Shape: (B * N,)

        # Convert points to PointCloud
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat_points.cpu().numpy()))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

        # Map points to their voxels
        flat_voxel_indices = np.array([voxel_grid.get_voxel(p.cpu().numpy()) for p in flat_points])
        flat_voxel_indices = torch.tensor(flat_voxel_indices, dtype=torch.int32, device=device)

        # Combine batch IDs with voxel indices
        combined_indices = torch.cat([batch_indices.unsqueeze(1), flat_voxel_indices], dim=1)

        # Group points by voxel
        unique_voxels, inverse_indices = torch.unique(combined_indices, dim=0, return_inverse=True)
        voxel_counts = torch.bincount(inverse_indices)

        # Compute voxel centroids
        voxel_sums = torch.zeros((unique_voxels.shape[0], 3), device=device).index_add_(
            0, inverse_indices, flat_points
        )
        voxel_centroids = voxel_sums / voxel_counts.unsqueeze(1)

        # Normalize points relative to voxel centroids
        norm_points = flat_points - voxel_centroids[inverse_indices]

        # Compute positional embeddings
        positional_embeddings = self.pos_enc_mlp(norm_points)

        # Concatenate input features (coordinates) and positional embeddings
        input_features = flat_points  # Shape: (B * N, 3)
        concat_features = torch.cat([input_features, positional_embeddings], dim=1)

        # Aggregate features per voxel by averaging
        aggregated_features = torch.zeros((unique_voxels.shape[0], concat_features.shape[1]), device=device).index_add_(
            0, inverse_indices, concat_features
        )
        aggregated_features /= voxel_counts.unsqueeze(1)
        aggregated_features = self.feature_mlp(aggregated_features)

        # Determine maximum number of voxels per batch for padding
        max_voxels = batch_indices.bincount().max().item()

        # Initialize padded tensors
        padded_aggregated_features = torch.zeros(
            (batch_size, max_voxels, aggregated_features.size(1)),
            device=device,
            dtype=aggregated_features.dtype,
        )
        padded_norm_points = torch.zeros(
            (batch_size, max_voxels, 3), device=device, dtype=norm_points.dtype
        )

        # Pad each batch
        for batch_id in range(batch_size):
            batch_voxel_mask = unique_voxels[:, 0] == batch_id
            batch_voxels = torch.where(batch_voxel_mask)[0]
            num_voxels = len(batch_voxels)

            # Pad the aggregated features and normalized points
            padded_aggregated_features[batch_id, :num_voxels] = aggregated_features[batch_voxels]
            padded_norm_points[batch_id, :num_voxels] = norm_points[batch_indices == batch_id][:num_voxels]

        return (
            padded_aggregated_features,
            padded_norm_points,
            voxel_centroids,
            voxel_counts,
            positional_embeddings,
            batch_indices.view(batch_size, -1),
        )
