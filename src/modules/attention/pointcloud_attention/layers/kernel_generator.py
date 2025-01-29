import torch

from modules.config.logger import Logger

class KernelGenerator:
    """
    A class for generating kernel offsets and related information based on 
    a given kernel size in a specified number of dimensions.

    This is primarily used for defining local neighborhoods in point cloud 
    self-attention models.
    """

    def __init__(self, kernel_size: int | tuple, dimension: int):
        """
        Initializes the KernelGenerator.

        Args:
            kernel_size (int or tuple):
                - If an integer is provided, the same kernel size is used for all dimensions.
                - If a tuple is provided, each element corresponds to the size along a specific dimension.
            dimension (int):
                - The number of dimensions in which the kernel operates.
        """
        
        # Initialize logger (class name passed as string for static usage if desired)
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"kernel_size={kernel_size}, dimension={dimension}")

        self.kernel_size = kernel_size
        self.dimension = dimension

        # Compute kernel volume (total number of elements in the kernel)
        self.kernel_volume = self._compute_kernel_volume()
        # self.logger.debug(f"Computed kernel_volume={self.kernel_volume}")

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
        # Example: for kernel_size=3, dimension=1, range -> [-1, 0, 1]
        if isinstance(self.kernel_size, int):
            half = self.kernel_size // 2
            ranges = [torch.arange(-half + 1, half + 1) for _ in range(self.dimension)]
        else:
            ranges = [torch.arange(-k // 2 + 1, k // 2 + 1) for k in self.kernel_size]

        # Create a meshgrid and flatten to get all offsets
        grid = torch.meshgrid(*ranges, indexing="ij")
        offsets = torch.stack(grid, dim=-1).view(-1, self.dimension)
        # self.logger.debug(f"Generated {offsets.shape[0]} kernel offsets.")

        return offsets

    def get_kernel_offsets(self) -> torch.Tensor:
        """
        Returns the precomputed kernel offsets.

        Returns:
            torch.Tensor: The kernel offsets of shape `[kernel_volume, dimension]`.
        """
        return self.kernel_offsets
