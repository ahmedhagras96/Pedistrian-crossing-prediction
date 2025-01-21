import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from modules.config.logger import Logger


class CentroidAwareVoxelization(nn.Module):
    """
    A module that voxelizes point cloud data and computes a feature representation 
    centered around voxel centroids. The class uses separate MLPs for positional 
    encoding and feature extraction.
    """

    def __init__(self, embed_dim: int):
        """
        Initialize the centroid-aware voxelization module.

        Args:
            embed_dim (int):
                Dimensionality of the embedding space for positional encoding
                and subsequent feature extraction.
        """
        super().__init__()
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(
            f"Initializing {self.__class__.__name__} with embed_dim={embed_dim}"
        )

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
            nn.Linear(embed_dim + 3, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # self.logger.debug(f"{self.__class__.__name__} successfully initialized.")

    def forward(self, points: torch.Tensor, voxel_size: float = 0.05):
        """
        Voxelizes the input point clouds, computes centroids and normalized positions,
        and aggregates features within each voxel.

        Args:
            points (torch.Tensor):
                Tensor of shape (B, N, 3) representing point coordinates,
                where B is the batch size and N is the number of points per batch.
            voxel_size (float, optional):
                Size of the voxel grid. Defaults to 0.05.

        Returns:
            padded_aggregated_features (torch.Tensor):
                Padded features of shape (B, max_voxels, embed_dim).
            padded_norm_points (torch.Tensor):
                Padded normalized points of shape (B, max_voxels, 3).
            voxel_centroids (torch.Tensor):
                Centroids of each voxel. Shape: (num_unique_voxels, 3).
            voxel_counts (torch.Tensor):
                Number of points per voxel. Shape: (num_unique_voxels,).
            pos_embs (torch.Tensor):
                Positional embeddings for the points. Shape: (B*N, embed_dim).
            batch_ids (torch.Tensor):
                Batch indices for each point, reshaped to (B, N).
        """
        # self.logger.debug(
        #     f"Forward called with points of shape {points.shape}, voxel_size={voxel_size}"
        # )

        batch_size, num_points, _ = points.shape
        device = points.device

        # 1) Flatten the batch dimension
        # ------------------------------
        flat_points = points.view(-1, 3)  # (B*N, 3)
        batch_ids = torch.arange(batch_size, device=device).repeat_interleave(num_points)

        # 2) Convert to Open3D PointCloud and create a voxel grid
        # -------------------------------------------------------
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat_points.cpu().numpy()))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

        # 3) Determine voxel indices for each point
        # -----------------------------------------
        flat_voxel_indices = np.array(
            [voxel_grid.get_voxel(p.cpu().numpy()) for p in flat_points]
        )
        flat_voxel_indices = torch.tensor(flat_voxel_indices, dtype=torch.int32, device=device)

        # Combine batch IDs with voxel indices for uniqueness across the batch dimension
        combined_indices = torch.cat([batch_ids.unsqueeze(1), flat_voxel_indices], dim=1)  # (B*N, 4)

        # Group points by voxel
        unique_voxels, inverse_indices = torch.unique(combined_indices, dim=0, return_inverse=True)
        voxel_counts = torch.bincount(inverse_indices)

        # self.logger.debug(
        #     f"Number of unique voxels: {unique_voxels.shape[0]}. "
        #     f"Total points: {flat_points.shape[0]}"
        # )

        # 4) Compute centroids for each voxel
        # -----------------------------------
        voxel_sums = torch.zeros(
            (unique_voxels.shape[0], 3), device=device
        ).index_add_(0, inverse_indices, flat_points)
        voxel_centroids = voxel_sums / voxel_counts.unsqueeze(1)  # (num_unique_voxels, 3)

        # 5) Normalize points relative to voxel centroids
        # ----------------------------------------------
        norm_points = flat_points - voxel_centroids[inverse_indices]

        # 6) Compute positional embeddings
        # --------------------------------
        pos_embs = self.pos_enc_mlp(norm_points)  # (B*N, embed_dim)

        # 7) Concatenate original coordinates with positional embeddings
        # --------------------------------------------------------------
        input_features = flat_points  # (B*N, 3)
        concat_features = torch.cat([input_features, pos_embs], dim=1)  # (B*N, embed_dim + 3)

        # 8) Aggregate features per voxel (average pooling)
        # -------------------------------------------------
        aggregated_features = torch.zeros(
            (unique_voxels.shape[0], concat_features.shape[1]), device=device
        ).index_add_(0, inverse_indices, concat_features)
        aggregated_features /= voxel_counts.unsqueeze(1)
        aggregated_features = self.feature_mlp(aggregated_features)  # (num_unique_voxels, embed_dim)

        # 9) Pad each batch to the maximum number of voxels
        # -------------------------------------------------
        max_voxels = batch_ids.bincount().max().item()
        padded_aggregated_features = torch.zeros(
            (batch_size, max_voxels, aggregated_features.size(1)),
            device=device, dtype=aggregated_features.dtype
        )
        padded_norm_points = torch.zeros(
            (batch_size, max_voxels, 3),
            device=device, dtype=norm_points.dtype
        )

        for b in range(batch_size):
            batch_voxel_mask = unique_voxels[:, 0] == b  # Identify voxels belonging to batch b
            batch_voxels = torch.where(batch_voxel_mask)[0]
            num_voxels = len(batch_voxels)

            # Identify points belonging to batch b
            batch_points_mask = batch_ids == b

            # Extract aggregated features and normalized points
            batch_aggregated_features = aggregated_features[batch_voxels]
            batch_norm_points = norm_points[batch_points_mask]

            # Pad
            padded_aggregated_features[b, :num_voxels] = batch_aggregated_features
            padded_norm_points[b, :num_voxels] = batch_norm_points[:num_voxels]

        # self.logger.debug(
        #     f"Finished voxelization and feature aggregation. "
        #     f"Output shapes: padded_aggregated_features={padded_aggregated_features.shape}, "
        #     f"padded_norm_points={padded_norm_points.shape}"
        # )

        # Reshape batch_ids back to (B, N) for downstream usage
        batch_ids_reshaped = batch_ids.view(batch_size, -1)

        return (
            padded_aggregated_features,
            padded_norm_points,
            voxel_centroids,
            voxel_counts,
            pos_embs,
            batch_ids_reshaped
        )
