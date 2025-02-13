import torch
import torch.nn as nn

from attention_vector.point_cloud_attn_vector.modules.kernel_generator import KernelGenerator

class LocalSelfAttentionBase(nn.Module):
    """
    A base class for local self-attention mechanisms that operate on sparse coordinates.
    Uses a `KernelGenerator` to define kernel offsets and constructs a kernel map
    for sparse tensor attention computations.
    """

    def __init__(self, kernel_size: int | tuple, dimension: int, sparse_ratio: float):
        """
        Initializes the base local self-attention module.

        Args:
            kernel_size (int or tuple):
                - If an integer is provided, the same kernel size is used for all dimensions.
                - If a tuple is provided, each element corresponds to the size along a specific dimension.
            dimension (int):
                - The number of spatial dimensions in which the kernel operates (e.g., 3 for 3D).
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.dimension = dimension

        # Initialize the kernel generator
        self.kernel_generator = KernelGenerator(kernel_size, dimension, sparse_ratio)
        self.kernel_volume = self.kernel_generator.kernel_volume  # Total number of elements in the kernel

    def get_kernel_map_and_out_key(self, sparse_coords: torch.sparse_coo_tensor):
        """
        Efficiently constructs a kernel mapping for sparse tensors using batch operations.

        Args:
            sparse_coords (torch.sparse_coo_tensor): Sparse coordinate tensor with shape (4, num_voxels).

        Returns:
            kernel_map (torch.Tensor): Tensor of shape (num_neighbors, 3) containing (input_idx, output_idx, rel_pos_idx).
            out_key_tensor (torch.Tensor): Tensor of unique output coordinates [M, D].
        """
        sparse_indices = sparse_coords.indices()  # Shape: (4, num_voxels)
        device = sparse_indices.device  
        num_voxels = sparse_indices.shape[1]

        # Extract batch IDs and spatial coordinates
        batch_ids = sparse_indices[0] 
        coordinates = sparse_indices[1:].T  

        # Move kernel offsets to the correct device
        kernel_offsets = self.kernel_generator.sample_sparse_kernel().to(device)

        # Coordinate Expansion 
        expanded_coords = coordinates.unsqueeze(1) + kernel_offsets.unsqueeze(0)  # Shape: (num_voxels, kernel_volume, 3)
        expanded_coords = expanded_coords.reshape(-1, self.dimension)

        # Batch Expansion
        batch_expanded = batch_ids.repeat_interleave(self.kernel_volume).reshape(-1, 1)  # (num_voxels * kernel_volume, 1)
        expanded_coords = expanded_coords.reshape(-1, self.dimension)  # (num_voxels * kernel_volume, 3)

        # Create Output Key Tensor
        out_key_tensor = torch.empty((num_voxels * self.kernel_volume, self.dimension + 1), device=device, dtype=torch.long)

        # Assign batch IDs
        out_key_tensor[:, 0] = batch_expanded.squeeze()

        # Ensure expanded_coords matches the expected size
        if expanded_coords.shape[0] != out_key_tensor.shape[0]:
            expanded_coords = expanded_coords.repeat(out_key_tensor.shape[0] // expanded_coords.shape[0] + 1, 1)
            expanded_coords = expanded_coords[:out_key_tensor.shape[0], :]

        if expanded_coords.shape[0] == out_key_tensor.shape[0]:
            out_key_tensor[:, 1:] = expanded_coords
        else:
            raise RuntimeError(f"Shape mismatch after correction: {expanded_coords.shape} vs {out_key_tensor.shape}")

        # Convert to Unique Voxel Keys
        _, inverse_indices = torch.unique(out_key_tensor, dim=0, return_inverse=True)

        # Map Output Indices Using Inverse Mapping
        output_indices = inverse_indices  # Shape: (num_voxels * kernel_volume,)

        # Final Kernel Map: (input_idx, output_idx, rel_pos_idx)
        input_indices = torch.arange(num_voxels, device=device).repeat_interleave(self.kernel_volume)
        rel_pos_indices = torch.arange(self.kernel_volume, device=device).repeat(num_voxels)  # Shape: (num_voxels * kernel_volume)

        kernel_map = torch.stack([input_indices, output_indices, rel_pos_indices], dim=1)  # Shape: (num_voxels * kernel_volume, 3)

        return kernel_map, out_key_tensor

    def key_query_map_from_kernel_map(self, kernel_map: torch.Tensor):
        """
        Converts kernel map into tensor-based mapping for GPU efficiency.

        Args:
            kernel_map (torch.Tensor): Tensor of shape [num_neighbors, 3] containing
                                      (input_idx, output_idx, rel_pos_idx).

        Returns:
            torch.Tensor: A structured key-query map of shape [num_neighbors, 3].
        """
        return kernel_map.clone()  
