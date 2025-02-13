import torch
import itertools


class KernelGenerator:
    """
    Generates sparse kernel offsets instead of a full neighborhood grid.
    """

    def __init__(self, kernel_size, dimension, sparse_ratio=0.5):
        """
        Initializes the KernelGenerator.

        Args:
            kernel_size (int or tuple): Kernel size (e.g., 3 for 3x3x3).
            dimension (int): Number of dimensions (usually 3 for 3D).
            sparse_ratio (float): Fraction of the full kernel to keep (e.g., 0.5 = 50% of neighbors).
        """
        self.kernel_size = kernel_size
        self.dimension = dimension
        self.sparse_ratio = sparse_ratio

        # Compute the full kernel
        self.full_kernel_offsets = self.generate_full_kernel_offsets()

        # Apply sparse sampling
        self.kernel_offsets = self.sample_sparse_kernel()

        self.kernel_volume = self._compute_kernel_volume()

    def _compute_kernel_volume(self) -> int:
        """
        Computes the volume of the kernel (i.e., the total number of elements).

        Returns:
            int: The number of elements in the kernel.
        """
        if isinstance(self.kernel_size, int):
            return self.kernel_size ** self.dimension
        return torch.prod(torch.tensor(self.kernel_size, dtype=torch.int32)).item()

    def generate_full_kernel_offsets(self):
        """Generates all possible kernel offsets in a memory-efficient way."""
        half = self.kernel_size // 2
        ranges = [list(range(-half, half + 1)) for _ in range(self.dimension)]

        # Generate Cartesian product (avoids full meshgrid allocation)
        offsets = torch.tensor(list(itertools.product(*ranges)), dtype=torch.int32)

        # Remove center voxel
        return offsets[~torch.all(offsets == 0, dim=1)]

    def sample_sparse_kernel(self):
        """Selects a subset of the full kernel more efficiently."""
        num_offsets = self.full_kernel_offsets.shape[0]
        num_samples = int(num_offsets * self.sparse_ratio)

        # Directly sample random indices (without full shuffle)
        sampled_indices = torch.randint(0, num_offsets, (num_samples,))
        return self.full_kernel_offsets[sampled_indices]


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
