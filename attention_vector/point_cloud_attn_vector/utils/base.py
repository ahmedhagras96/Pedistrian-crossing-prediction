import torch
import torch.nn as nn

class KernelGenerator:
    """
    A class for generating kernel offsets and related information based on 
    a given kernel size in a specified number of dimensions.

    This is primarily used for defining local neighborhoods in point cloud 
    self-attention models.
    """

    def __init__(self, kernel_size: int  | tuple, dimension: int):
        """
        Initializes the KernelGenerator.

        Args:
            kernel_size (int or tuple):
                - If an integer is provided, the same kernel size is used for all dimensions.
                - If a tuple is provided, each element corresponds to the size along a specific dimension.
            dimension (int):
                - The number of dimensions in which the kernel operates.
        """
        self.kernel_size = kernel_size
        self.dimension = dimension

        # Compute kernel volume (total number of elements in the kernel)
        self.kernel_volume = self._compute_kernel_volume()

        # Precompute kernel offsets
        self.kernel_offsets = self._generate_kernel_offsets()

    def _compute_kernel_volume(self) -> int:
        """
        Computes the volume of the kernel (i.e., the total number of elements).

        Returns:
            int: The number of elements in the kernel.
        """
        if isinstance(self.kernel_size, int):
            return self.kernel_size ** self.dimension
        return torch.prod(torch.tensor(self.kernel_size, dtype=torch.int32)).item()

    def _generate_kernel_offsets(self) -> torch.Tensor:
        """
        Generates relative offsets for each element in the kernel.

        Returns:
            torch.Tensor: A tensor of shape `[kernel_volume, dimension]`, 
                          where each row represents an offset.
        """
        # Generate an integer range centered around 0 for each dimension
        if isinstance(self.kernel_size, int):
            half = self.kernel_size // 2
            ranges = [torch.arange(-half + 1, half + 1) for _ in range(self.dimension)]
        else:
            ranges = [torch.arange(-k // 2 + 1, k // 2 + 1) for k in self.kernel_size]

        # Create a meshgrid and flatten to get all offsets
        grid = torch.meshgrid(*ranges, indexing="ij")
        offsets = torch.stack(grid, dim=-1).view(-1, self.dimension)

        return offsets

    def get_kernel_offsets(self) -> torch.Tensor:
        """
        Returns the precomputed kernel offsets.

        Returns:
            torch.Tensor: The kernel offsets of shape `[kernel_volume, dimension]`.
        """
        return self.kernel_offsets


class LocalSelfAttentionBase(nn.Module):
    """
    A base class for local self-attention mechanisms that operate on sparse coordinates.
    Uses a `KernelGenerator` to define kernel offsets and constructs a kernel map
    for sparse tensor attention computations.
    """

    def __init__(self, kernel_size: int | tuple, dimension: int):
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
        self.kernel_generator = KernelGenerator(kernel_size, dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume  # Total number of elements in the kernel

    def get_kernel_map_and_out_key(self, sparse_coords: torch.sparse_coo_tensor):
        """
        Constructs a kernel mapping for sparse tensors.

        This function maps each voxel to its surrounding neighborhood based on 
        the kernel offsets, generating a mapping between input and output indices.

        Args:
            sparse_coords (torch.sparse_coo_tensor): 
                - Sparse coordinate tensor of shape `(4, num_voxels)`, where:
                  - The first row represents batch indices.
                  - The remaining rows represent spatial coordinates (x, y, z).

        Returns:
            tuple:
                - **kernel_map** (list of tuples): Each tuple `(input_idx, output_idx, rel_pos_idx)`
                  represents the mapping of input indices to output indices.
                - **out_key_tensor** (torch.Tensor): A tensor of shape `[M, D]` representing 
                  unique output coordinates for the sparse tensor.
        """
        sparse_indices = sparse_coords.indices()  # Shape: (4, num_voxels)
        num_voxels = sparse_indices.shape[1]

        kernel_map = []
        out_key = []
        key_to_index = {}
        current_index = 0

        # Extract batch IDs and spatial coordinates
        batch_ids = sparse_indices[0]  # First row contains batch indices
        coordinates = sparse_indices[1:].T  # Extract (x, y, z) indices

        # Iterate over all points in the sparse tensor
        for i in range(num_voxels):
            batch_id = batch_ids[i].item()
            coord = coordinates[i]

            for rel_pos_idx, offset in enumerate(self.kernel_generator.kernel_offsets):
                neighbor = tuple((coord + offset).tolist())
                key = (batch_id, neighbor)

                # If this neighbor hasn't been encountered before, add it
                if key not in key_to_index:
                    key_to_index[key] = current_index
                    out_key.append(key)
                    current_index += 1

                output_index = key_to_index[key]
                kernel_map.append((i, output_index, rel_pos_idx))

        # Convert the output key list to a tensor
        out_key_tensor = torch.tensor(
            [key[1] for key in out_key],
            device=sparse_coords.device,
            dtype=torch.long
        )

        return kernel_map, out_key_tensor

    def key_query_map_from_kernel_map(self, kernel_map: list):
        """
        Converts a kernel map to a key-query mapping.

        This function reshapes and labels the same data structure to explicitly
        represent key-query relationships.

        Args:
            kernel_map (list of tuples): 
                - Each tuple `(input_idx, output_idx, rel_pos_idx)` defines 
                  the mapping between input indices and their corresponding 
                  neighborhood outputs.

        Returns:
            list of tuples: A structured key-query map in the same format.
        """
        return [(input_idx, output_idx, rel_pos_idx) for input_idx, output_idx, rel_pos_idx in kernel_map]