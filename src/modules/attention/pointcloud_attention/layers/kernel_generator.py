import torch

from modules.config.logger import Logger


class KernelGenerator:
    """
    Generates kernel offsets and related information for a given kernel size
    in a specified number of dimensions.
    """

    def __init__(self, kernel_size, dimension):
        """
        Initialize the KernelGenerator.

        Args:
            kernel_size (int or tuple): Size of the kernel. If an integer is provided, it is
                used for all dimensions; if a tuple is provided, each element corresponds to
                the size along a specific dimension.
            dimension (int): The number of dimensions in which the kernel will operate.
        """
        # Initialize logger (class name passed as string for static usage if desired)
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"kernel_size={kernel_size}, dimension={dimension}")

        self.kernel_size = kernel_size
        self.dimension = dimension

        # Compute kernel volume (number of elements in the kernel)
        if isinstance(kernel_size, int):
            self.kernel_volume = kernel_size ** dimension
        else:
            self.kernel_volume = torch.prod(
                torch.tensor(kernel_size, dtype=torch.int32)
            ).item()
        # self.logger.debug(f"Computed kernel_volume={self.kernel_volume}")

        # Precompute offsets for the kernel
        self.kernel_offsets = self.generate_kernel_offsets()

    def generate_kernel_offsets(self) -> torch.Tensor:
        """
        Generate relative offsets for each element in the kernel.

        Returns:
            torch.Tensor: A tensor of shape [kernel_volume, dimension], where
                each row represents an offset.
        """
        # For each dimension, generate an integer range centered around 0
        # Example: for kernel_size=3, dimension=1, range -> [-1, 0, 1]
        if isinstance(self.kernel_size, int):
            half = self.kernel_size // 2
            ranges = [
                torch.arange(-half + 1, half + 1) for _ in range(self.dimension)
            ]
        else:
            # If kernel_size is a tuple, handle each dimension's range independently
            ranges = []
            for k in self.kernel_size:
                half = k // 2
                ranges.append(torch.arange(-half + 1, half + 1))

        # Create a meshgrid and flatten to get all offsets
        grid = torch.meshgrid(*ranges, indexing="ij")
        offsets = torch.stack(grid, dim=-1).view(-1, self.dimension)

        # self.logger.debug(f"Generated {offsets.shape[0]} kernel offsets.")
        return offsets
