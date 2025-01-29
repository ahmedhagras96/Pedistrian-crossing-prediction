import torch
import torch.nn as nn
import open3d.ml.torch as o3dml


class CentroidAwareVoxelization(nn.Module):
    """
    Centroid-aware voxelization for point cloud data, integrating voxel pooling,
    normalization, and sparse feature representation.

    This module performs:
    1. **Voxelization** of point clouds.
    2. **Feature pooling** within voxels.
    3. **Normalization** using voxel centroids.
    4. **Sparse tensor representation** for efficient processing.
    """

    def __init__(self, embed_dim: int, max_voxel_grid_size: int = int(1e5)):
        """
        Initializes the CentroidAwareVoxelization module.

        Args:
            embed_dim (int):
                The embedding dimension for learned features.
            max_voxel_grid_size (int, optional):
                Maximum voxel grid size for sparse tensor representation. Defaults to `int(1e5)`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_voxel_grid_size = max_voxel_grid_size

        # Positional encoding MLP
        self.pos_enc_mlp = nn.Sequential(
            nn.Linear(3, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Feature aggregation MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim + 6, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    @staticmethod
    def pad_tensor(tensor: torch.Tensor, target_size: int, shape: tuple, fill_value=0, device="cpu"):
        """
        Pads a tensor to match the target size.

        Args:
            tensor (torch.Tensor): Input tensor to be padded.
            target_size (int): The target number of elements in dimension 0.
            shape (tuple): The shape of each element (excluding batch size).
            fill_value (int, optional): Value used for padding. Defaults to 0.
            device (str, optional): Device to place the padded tensor. Defaults to "cpu".

        Returns:
            torch.Tensor: Padded tensor of shape `(target_size, *shape)`.
        """
        pad_size = target_size - tensor.shape[0]
        if pad_size > 0:
            padding = torch.full((pad_size, *shape), fill_value, dtype=tensor.dtype, device=device)
            return torch.cat([tensor, padding], dim=0)
        return tensor

    def forward(self, points: torch.Tensor, voxel_size: float = 0.05):
        """
        Forward pass for centroid-aware voxelization.

        Args:
            points (torch.Tensor): 
                Input point cloud of shape `(B, N, 3)`, where:
                - `B`: Batch size
                - `N`: Number of points per batch
                - `3`: (x, y, z) coordinates
                
            voxel_size (float, optional): 
                The size of each voxel. Defaults to 0.05.

        Returns:
            tuple: 
                - **sparse_features** (torch.sparse.Tensor): Sparse representation of voxelized features.
                - **norm_points** (torch.Tensor): Normalized points.
                - **voxel_centroids** (torch.Tensor): Centroids of voxels.
                - **voxel_counts** (torch.Tensor): Number of points per voxel.
                - **pos_embs** (torch.Tensor): Encoded positional embeddings.
        """

        batch_size, num_points, _ = points.shape
        device = points.device

        # 🔹 **Flatten points and prepare row_splits**
        flat_points = points.view(-1, 3)
        row_splits = torch.arange(0, (batch_size + 1) * num_points, num_points, device=device, dtype=torch.int64)

        # 🔹 **Compute spatial range**
        points_range_min = flat_points.min(dim=0).values
        points_range_max = flat_points.max(dim=0).values

        # 🔹 **Voxelization**
        voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits = o3dml.ops.voxelize(
            points=flat_points,
            row_splits=row_splits,
            voxel_size=torch.tensor([voxel_size] * 3, device=device),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            max_points_per_voxel=1000
        )

        # 🔹 **Compute unique voxel indices per batch**
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            voxel_batch_splits[1:] - voxel_batch_splits[:-1]
        ).unsqueeze(1)

        unique_voxels = torch.cat([batch_indices, voxel_coords], dim=1)  # Shape: (num_voxels, 4)

        # 🔹 **Voxel pooling**
        voxel_avg_positions, pooled_features = o3dml.ops.voxel_pooling(
            positions=flat_points,
            features=flat_points,
            voxel_size=voxel_size,
            position_fn='average',
            feature_fn='average'
        )

        # 🔹 **Compute voxel point counts**
        voxel_counts = (voxel_point_row_splits[1:] - voxel_point_row_splits[:-1]).unsqueeze(1).float()

        # 🔹 **Handle tensor mismatch issues**
        min_voxels = min(voxel_avg_positions.shape[0], voxel_counts.shape[0])
        target_size = batch_size * num_points

        voxel_avg_positions = self.pad_tensor(voxel_avg_positions[:min_voxels], target_size, (3,), device=device)
        voxel_counts = self.pad_tensor(voxel_counts[:min_voxels], target_size, (1,), device=device)
        pooled_features = self.pad_tensor(pooled_features[:min_voxels], target_size, (3,), device=device)
        unique_voxels = self.pad_tensor(unique_voxels[:min_voxels], target_size, (4,), fill_value=-1, device=device)
        voxel_point_indices = self.pad_tensor(voxel_point_indices[:min_voxels], target_size, (), fill_value=-1, device=device)
        flat_points = self.pad_tensor(flat_points[:min_voxels], target_size, (3,), device=device)

        # 🔹 **Compute voxel centroids**
        voxel_sums = voxel_avg_positions * voxel_counts
        voxel_centroids = voxel_sums / (voxel_counts + 1e-6)

        # Ensure voxel_point_indices is within valid range
        voxel_point_indices = voxel_point_indices.clamp(0, voxel_centroids.shape[0] - 1)

        #🔹 **Normalize points**
        norm_points = flat_points - voxel_centroids[voxel_point_indices]

        # 🔹 **Feature aggregation**
        pos_embs = self.pos_enc_mlp(norm_points)
        concat_features = torch.cat([flat_points, pos_embs, pooled_features], dim=1)
        aggregated_features = self.feature_mlp(concat_features)

        # 🔹 **Sparse tensor conversion**
        transposed_unique_voxels = unique_voxels.permute(1, 0)
        sparse_features = torch.sparse_coo_tensor(
            indices=transposed_unique_voxels,
            values=aggregated_features,
            size=(batch_size, self.max_voxel_grid_size, self.max_voxel_grid_size, self.max_voxel_grid_size, self.embed_dim),
            device=device, 
            requires_grad=True
        )

        return (
            sparse_features,
            norm_points,
            voxel_centroids,
            voxel_counts,
            pos_embs,
        )
