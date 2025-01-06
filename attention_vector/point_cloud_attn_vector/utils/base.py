import torch
import torch.nn as nn

class KernelGenerator:
    def __init__(self, kernel_size, stride, dilation, dimension):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        if isinstance(kernel_size, int):
            self.kernel_volume = kernel_size ** dimension
        else:
            self.kernel_volume = torch.prod(torch.tensor(kernel_size, dtype=torch.int32)).item()
        self.kernel_offsets = self.generate_kernel_offsets()

    def generate_kernel_offsets(self):
        ranges = [
            torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
            for _ in range(self.dimension)
        ]
        grid = torch.meshgrid(*ranges, indexing="ij")
        offsets = torch.stack(grid, dim=-1).view(-1, self.dimension)
        return offsets

    def get_kernel(self, tensor_stride, is_transpose=False):
        region_type = "cube"  # Assuming a cube region type for simplicity
        region_offset = self.kernel_offsets
        return region_type, region_offset, tensor_stride

class LocalSelfAttentionBase(nn.Module):
    def __init__(self, kernel_size, stride, dilation, dimension):
        super(LocalSelfAttentionBase, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        self.kernel_generator = KernelGenerator(kernel_size, stride, dilation, dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

    def get_kernel_map_and_out_key(self, coordinates, batch_indices):
        """
        Args:
            coordinates: Tensor of shape [B, N, D] where D=3
            batch_indices: Tensor of shape [B, N] indicating the batch index for each point
        Returns:
            kernel_map: list of tuples (input_index, output_index, relative_position_index)
            out_key_tensor: Tensor of shape [M, D] representing unique output coordinates
        """
        kernel_map = []
        out_key = []
        key_to_index = {}
        current_index = 0

        B = batch_indices.max().item() + 1
        for b in range(B):
            batch_coords = coordinates[batch_indices == b]  # [N_b, D]
            for idx, coord in enumerate(batch_coords):
                for rel_pos_idx, offset in enumerate(self.kernel_generator.kernel_offsets):
                    neighbor = tuple((coord + offset).tolist())
                    key = (b, neighbor)
                    if key not in key_to_index:
                        key_to_index[key] = current_index
                        out_key.append(key)
                        current_index += 1
                    output_index = key_to_index[key]
                    kernel_map.append((idx, output_index, rel_pos_idx))

        out_key_tensor = torch.tensor(
            [key[1] for key in out_key], device=coordinates.device, dtype=torch.long
        )
        return kernel_map, out_key_tensor

    def key_query_map_from_kernel_map(self, kernel_map):
        kq_map = []
        for input_idx, output_idx, rel_pos_idx in kernel_map:
            kq_map.append((
                input_idx,            
                output_idx,           
                rel_pos_idx           
            ))
        return kq_map

    def key_query_indices_from_kernel_map(self, kq_map):
        indices = []
        for input_indices, output_index, rel_pos_idx in kq_map:
            indices.append((input_indices, output_index, rel_pos_idx))
        return indices