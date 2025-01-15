import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import open3d as o3d
import numpy as np

class CentroidAwareVoxelization(nn.Module):
    def __init__(self, embed_dim):
        super(CentroidAwareVoxelization, self).__init__()
        self.embed_dim = embed_dim

        # Shared positional embedding MLP
        self.pos_enc_mlp = nn.Sequential(
            nn.Linear(3, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Voxel feature generator MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim+3, embed_dim, bias=False),  
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, voxel_size=0.05):
        """
        Compute voxelized representation and features from point cloud data.

        Args:
            points (torch.Tensor): Tensor of shape (B, N, 3) representing point coordinates.
            voxel_size (float): Size of the voxel grid.

        Returns:
            padded_aggregated_features (torch.Tensor): Padded features of shape (B, max_voxels, embed_dim).
            padded_norm_points (torch.Tensor): Padded normalized points of shape (B, max_voxels, 3).
            voxel_centroids (torch.Tensor): Centroids of each voxel.
            voxel_counts (torch.Tensor): Number of points in each voxel.
            pos_embs (torch.Tensor): Positional embeddings for the points.
            batch_ids (torch.Tensor): Batch indices for each point.
        """

        batch_size, num_points, _ = points.shape
        device = points.device

        # Flatten batch dimension
        flat_points = points.view(-1, 3)  # Shape: (B * N, 3)
        batch_ids = torch.arange(batch_size, device=device).repeat_interleave(num_points)  # Shape: (B * N,)

        # Convert points to PointCloud
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat_points.cpu().numpy()))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

        # Map points to their voxels
        flat_voxel_indices = np.array([voxel_grid.get_voxel(p.cpu().numpy()) for p in flat_points])
        flat_voxel_indices = torch.tensor(flat_voxel_indices, dtype=torch.int32, device=device)

        # Combine batch IDs with voxel indices for uniqueness across batches
        combined_indices = torch.cat([batch_ids.unsqueeze(1), flat_voxel_indices], dim=1)  # Shape: (B * N, 4)

        # Group points by voxel
        unique_voxels, inverse_indices = torch.unique(combined_indices, dim=0, return_inverse=True)
        voxel_counts = torch.bincount(inverse_indices)

        # Compute centroids for each voxel
        voxel_sums = torch.zeros((unique_voxels.shape[0], 3), device=device).index_add_(
            0, inverse_indices, flat_points
        )
        voxel_centroids = voxel_sums / voxel_counts.unsqueeze(1)

        # Normalize points relative to voxel centroids
        norm_points = flat_points - voxel_centroids[inverse_indices]

        # Compute positional embeddings
        pos_embs = self.pos_enc_mlp(norm_points)

        # Concatenate input features (coordinates) and positional embeddings
        input_features = flat_points  # Shape: (B * N, 3)
        concat_features = torch.cat([input_features, pos_embs], dim=1)

        # Aggregate features per voxel by averaging
        aggregated_features = torch.zeros((unique_voxels.shape[0], concat_features.shape[1]), device=device).index_add_(
            0, inverse_indices, concat_features
        )
        aggregated_features /= voxel_counts.unsqueeze(1)
        aggregated_features = self.feature_mlp(aggregated_features)


        # Determine maximum number of voxels per batch for padding
        max_voxels = batch_ids.bincount().max().item()
        # Initialize padded tensors
        padded_aggregated_features = torch.zeros((batch_size, max_voxels, aggregated_features.size(1)),
                                                device=device, dtype=aggregated_features.dtype)
        padded_norm_points = torch.zeros((batch_size, max_voxels, 3), device=device, dtype=norm_points.dtype)

        # Pad each batch
        for b in range(batch_size):
            # Get voxel indices for this batch
            batch_voxel_mask = unique_voxels[:, 0] == b  # Unique voxels for this batch
            batch_voxels = torch.where(batch_voxel_mask)[0]
            num_voxels = len(batch_voxels)

            # Map inverse_indices to the batch
            batch_points_mask = batch_ids == b
            batch_aggregated_features = aggregated_features[batch_voxels]
            batch_norm_points = norm_points[batch_points_mask]

            # Pad the aggregated features and normalized points
            padded_aggregated_features[b, :num_voxels] = batch_aggregated_features
            padded_norm_points[b, :num_voxels] = batch_norm_points[:num_voxels]

        return padded_aggregated_features, padded_norm_points, voxel_centroids, voxel_counts, pos_embs, batch_ids.view(batch_size,-1)
