import torch
import torch.nn as nn

from modules.attention.pointcloud_attention.layers.kernel_generator import KernelGenerator
from modules.config.logger import Logger


class LocalSelfAttentionBase(nn.Module):
    """
    A base class for local self-attention mechanisms that operate on coordinates.
    It uses a KernelGenerator to create kernel offsets, then builds a kernel map
    that can be used in attention computations.
    """

    def __init__(self, kernel_size, dimension):
        """
        Initialize the base local self-attention module.

        Args:
            kernel_size (int or tuple): Size of the kernel in each dimension.
            dimension (int): The dimensionality of the coordinate space (e.g., 3 for 3D).
        """
        super().__init__()
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"kernel_size={kernel_size}, dimension={dimension}")

        self.kernel_size = kernel_size
        self.dimension = dimension

        # Create the kernel generator
        self.kernel_generator = KernelGenerator(kernel_size, dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

        # self.logger.debug(f"Kernel volume set to {self.kernel_volume}")

    def get_kernel_map_and_out_key(self, coordinates: torch.Tensor, batch_indices: torch.Tensor):
        """
        Build a map of input indices to output indices based on local neighborhoods.

        Args:
            coordinates (torch.Tensor):
                A tensor of shape [B, N, D], where B is total batch size (flattened),
                N is number of points in that batch, and D is the coordinate dimension.
            batch_indices (torch.Tensor):
                A tensor of shape [B, N] (or possibly [B*N]) indicating the batch index
                for each point.

        Returns:
            kernel_map (list of tuples):
                Each tuple is (input_index, output_index, rel_pos_idx),
                where 'rel_pos_idx' indexes into self.kernel_generator.kernel_offsets.
            out_key_tensor (torch.Tensor):
                A tensor of shape [M, D] representing the unique output coordinates
                for the entire set of points.
        """
        # self.logger.debug(f"Building kernel map for coordinates of shape {coordinates.shape}")

        kernel_map = []
        out_key = []
        key_to_index = {}
        current_index = 0

        # Number of batches is max index + 1
        B = batch_indices.max().item() + 1
        # self.logger.debug(f"Detected {B} distinct batch(es).")

        # Build kernel map for each batch
        for b in range(B):
            # Extract coordinates belonging to the current batch 'b'
            batch_coords = coordinates[batch_indices == b]

            for idx, coord in enumerate(batch_coords):
                # For each offset in the kernel, compute neighbor coordinate
                for rel_pos_idx, offset in enumerate(self.kernel_generator.kernel_offsets):
                    neighbor = tuple((coord + offset).tolist())
                    key = (b, neighbor)

                    # If this neighbor hasn't been encountered, add it
                    if key not in key_to_index:
                        key_to_index[key] = current_index
                        out_key.append(key)
                        current_index += 1

                    # Retrieve the output index for this neighbor
                    output_index = key_to_index[key]

                    # Append mapping info
                    kernel_map.append((idx, output_index, rel_pos_idx))

        # Convert the out_key (which contains (batch, coordinate)) to a tensor of just coordinates
        out_key_tensor = torch.tensor(
            [key[1] for key in out_key],
            device=coordinates.device,
            dtype=torch.long
        )

        # self.logger.debug(f"Kernel map size: {len(kernel_map)}; out_key_tensor shape: {out_key_tensor.shape}")
        return kernel_map, out_key_tensor

    def key_query_map_from_kernel_map(self, kernel_map):
        """
        Convert a kernel map to a key-query map. This is effectively
        a reshaping or labeling of the same indices.

        Args:
            kernel_map (list of tuples): Each tuple is (input_index, output_index, rel_pos_idx).

        Returns:
            list of tuples: Each tuple is (input_idx, output_idx, rel_pos_idx),
            representing the same data with a more explicit naming scheme.
        """
        # self.logger.debug(f"Building key-query map from kernel map of length {len(kernel_map)}")

        kq_map = []
        for input_idx, output_idx, rel_pos_idx in kernel_map:
            kq_map.append((input_idx, output_idx, rel_pos_idx))

        return kq_map

    def key_query_indices_from_kernel_map(self, kq_map):
        """
        Extract the indices from a key-query map for further processing.

        Args:
            kq_map (list of tuples): Each tuple is (input_idx, output_idx, rel_pos_idx).

        Returns:
            list of tuples: Each tuple is (input_idx, output_idx, rel_pos_idx).
        """
        # self.logger.debug(f"Extracting key-query indices from map of length {len(kq_map)}")

        indices = []
        for input_idx, output_idx, rel_pos_idx in kq_map:
            indices.append((input_idx, output_idx, rel_pos_idx))

        return indices
