import torch
import torch.nn as nn

from modules.attention.point_cloud_attention.kernel import KernelGenerator


class LocalSelfAttention(nn.Module):
    """
    Base class for local self-attention mechanisms using kernel-based operations.

    Args:
        kernel_size (int or tuple): Size of the kernel.
        stride (int): Stride for kernel application.
        dilation (int): Dilation factor for kernel elements.
        dimension (int): Dimensionality of the kernel.

    Attributes:
        kernel_size (int or tuple): Size of the kernel.
        stride (int): Stride for kernel application.
        dilation (int): Dilation factor.
        dimension (int): Dimensionality of the kernel.
        kernel_generator (KernelGenerator): Instance for generating kernel properties.
        kernel_volume (int): Total number of elements in the kernel.
    """

    def __init__(self, kernel_size, stride, dilation, dimension):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension

        self.kernel_generator = KernelGenerator(kernel_size, dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

    def compute_kernel_mapping(self, coordinates, batch_indices):
        """
        Computes kernel mapping and unique output coordinates.

        Args:
            coordinates (torch.Tensor): Tensor of shape [B, N, D] representing input coordinates.
            batch_indices (torch.Tensor): Tensor of shape [B, N] indicating batch indices.

        Returns:
            tuple: A tuple containing:
                - kernel_map (list): List of tuples (input_idx, output_idx, rel_pos_idx).
                - output_key_tensor (torch.Tensor): Unique output coordinates of shape [M, D].
        """
        kernel_map = []
        output_keys = []
        key_to_index = {}
        current_index = 0

        num_batches = batch_indices.max().item() + 1
        for batch_id in range(num_batches):
            batch_coords = coordinates[batch_indices == batch_id]  # Shape: [N_b, D]
            for input_idx, coord in enumerate(batch_coords):
                for rel_pos_idx, offset in enumerate(self.kernel_generator.kernel_offsets):
                    neighbor_coord = tuple((coord + offset).tolist())
                    key = (batch_id, neighbor_coord)
                    if key not in key_to_index:
                        key_to_index[key] = current_index
                        output_keys.append(key)
                        current_index += 1
                    output_idx = key_to_index[key]
                    kernel_map.append((input_idx, output_idx, rel_pos_idx))

        output_key_tensor = torch.tensor(
            [key[1] for key in output_keys], device=coordinates.device, dtype=torch.long
        )
        return kernel_map, output_key_tensor

    def map_key_query_indices(self, kernel_map):
        """
        Maps key-query relationships based on the kernel map.

        Args:
            kernel_map (list): Kernel map containing (input_idx, output_idx, rel_pos_idx).

        Returns:
            list: List of tuples (input_idx, output_idx, rel_pos_idx).
        """
        return [
            (input_idx, output_idx, rel_pos_idx) for input_idx, output_idx, rel_pos_idx in kernel_map
        ]
