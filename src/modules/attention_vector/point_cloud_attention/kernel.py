import torch
from modules.utilities.logger import LoggerUtils


class KernelGenerator:
    """
    Utility class to generate kernel offsets and compute kernel-related properties.

    Args:
        kernel_size (int or tuple): Size of the kernel (e.g., 3 for a 3x3 kernel).
        dimension (int): Dimensionality of the kernel (e.g., 3 for 3D space).

    Attributes:
        kernel_size (int or tuple): Size of the kernel.
        dimension (int): Dimensionality of the kernel.
        kernel_volume (int): Total number of elements in the kernel.
        kernel_offsets (torch.Tensor): Offsets of all kernel elements relative to the center.
    """

    def __init__(self, kernel_size, dimension):
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

        self.kernel_size = kernel_size
        self.dimension = dimension

        if isinstance(kernel_size, int):
            self.kernel_volume = kernel_size ** dimension
        else:
            self.kernel_volume = torch.prod(torch.tensor(kernel_size, dtype=torch.int32)).item()

        self.kernel_offsets = self._generate_kernel_offsets()

    def _generate_kernel_offsets(self):
        """
        Generates relative offsets for all elements in the kernel.

        Returns:
            torch.Tensor: Tensor of shape [kernel_volume, dimension], where each row represents
                          an offset in the kernel.
        """
        self.logger.debug("Generating kernel offsets.")
        ranges = [
            torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
            for _ in range(self.dimension)
        ]
        grid = torch.meshgrid(*ranges, indexing="ij")
        offsets = torch.stack(grid, dim=-1).view(-1, self.dimension)
        return offsets

    def get_kernel_region(self, tensor_stride):
        """
        Retrieves kernel region properties.

        Args:
            tensor_stride (int): Stride of the tensor.

        Returns:
            tuple: A tuple containing:
                - region_type (str): Type of kernel region ("cube" in this implementation).
                - region_offsets (torch.Tensor): Offsets of kernel elements.
                - tensor_stride (int): Tensor stride.
        """
        region_type = "cube"
        region_offsets = self.kernel_offsets
        return region_type, region_offsets, tensor_stride
