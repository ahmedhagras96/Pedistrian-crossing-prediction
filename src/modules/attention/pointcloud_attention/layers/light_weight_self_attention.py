import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention.pointcloud_attention.layers.local_self_attention_base import LocalSelfAttentionBase
from modules.config.logger import Logger

class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    """
    A lightweight self-attention layer for sparse point cloud data, operating in 3D space.
    This layer processes sparse tensors efficiently by leveraging kernel-based neighborhood computations.

    Inherits from `LocalSelfAttentionBase` for kernel operations.
    """

    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 3, num_heads: int = 4):
        """
        Initializes the LightweightSelfAttentionLayer.

        Args:
            in_channels (int): 
                Number of input feature channels per point.
            out_channels (int, optional): 
                Number of output feature channels after attention. Defaults to in_channels.
            kernel_size (int, optional): 
                Size of the local attention kernel. Defaults to 3.
            num_heads (int, optional): 
                Number of attention heads. Defaults to 4.
        """
        super().__init__(kernel_size=kernel_size, dimension=3)

        # Create a logger for this class
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with "
                         f"in_channels={in_channels}, out_channels={out_channels}, "
                         f"kernel_size={kernel_size}, num_heads={num_heads}")

        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        # Store layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        # Linear transformations for queries, values, and output
        self.to_query = nn.Linear(in_channels, out_channels, bias=False)
        self.to_value = nn.Linear(in_channels, out_channels, bias=False)
        self.to_out = nn.Linear(out_channels, out_channels, bias=False)

        # Absolute (intra) positional encoding
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels, bias=False),
        )

        # Relative (inter) positional encoding
        self.inter_pos_enc = nn.Parameter(
            torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels)
        )
        nn.init.normal_(self.inter_pos_enc, mean=0.0, std=1.0)

        # Global feature aggregation using adaptive max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, sparse_x: torch.sparse_coo_tensor, norm_points: torch.Tensor):
        """
        Forward pass for lightweight self-attention on sparse point cloud data.

        Args:
            sparse_x (torch.sparse_coo_tensor): 
                Sparse input feature tensor.
            norm_points (torch.Tensor): 
                Normalized point coordinates of shape `[B, N, 3]`.

        Returns:
            tuple:
                - **out** (torch.Tensor): Aggregated output features of shape `[B, out_channels]`.
                - **attn** (torch.Tensor): Attention map of shape `[B, M, num_heads]`.
        """
        # Extract sparse tensor components
        sparse_indices = sparse_x.indices()  # Shape: (4, num_voxels)
        sparse_values = sparse_x.values()  # Shape: (num_voxels, in_channels)
        
        # self.logger.debug(f"Forward called with sparse_x of shape {sparse_x.shape}, "
        #                   f"norm_points of shape {norm_points.shape}.")

        # 🔹 Determine batch size
        B = sparse_indices[0].max().item() + 1  

        # 🔹 Apply Absolute Positional Encoding (Intra-Position Encoding)
        intra_pos_enc = self.intra_pos_mlp(norm_points.view(-1, 3))  # [B*N, in_channels]
        sparse_values += intra_pos_enc[sparse_indices[0]]  # Add intra-positional encoding

        # 🔹 Compute Queries, Keys, and Values
        q = self.to_query(sparse_values).view(-1, self.num_heads, self.attn_channels)  # [num_voxels, num_heads, attn_channels]
        v = self.to_value(sparse_values).view(-1, self.num_heads, self.attn_channels)  # [num_voxels, num_heads, attn_channels]

        # 🔹 Kernel Mapping for Neighborhood Indices
        kernel_map, out_coordinates = self.get_kernel_map_and_out_key(sparse_x)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        device = sparse_x.device
        dtype = sparse_x.dtype

        # 🔹 Prepare Tensors for Sparse Attention Computation
        input_indices = torch.tensor([kq[0] for kq in kq_map], device=device, dtype=torch.long)
        output_indices = torch.tensor([kq[1] for kq in kq_map], device=device, dtype=torch.long)
        rel_pos_indices = torch.tensor([kq[2] for kq in kq_map], device=device, dtype=torch.long)

        # 🔹 Compute Maximum Output Points (M)
        M = max(out_coordinates.shape[0], output_indices.shape[0])
        attn = torch.zeros((B, M, self.num_heads), device=device, dtype=dtype)

        # 🔹 Ensure `output_indices` Matches `M`
        output_indices = output_indices.clamp(0, M - 1)
        num_values = min(output_indices.shape[0], M)
        output_indices = output_indices[:num_values]

        # 🔹 Normalize Queries & Compute Cosine Similarity
        norm_q = F.normalize(q, p=2, dim=-1)  
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)  

        q_gathered = norm_q[input_indices]  
        pos_enc_gathered = norm_pos_enc[rel_pos_indices]  

        attn_contribution = (q_gathered * pos_enc_gathered.unsqueeze(0)).sum(dim=-1)  
        attn_contribution = attn_contribution.expand(B, -1, -1)

        # 🔹 Expand Output Indices for Scatter Add
        expanded_output_indices = output_indices.unsqueeze(0).expand(B, -1)

        # 🔹 Compute Sparse Attention Weights
        attn.scatter_add_(
            1,
            expanded_output_indices.unsqueeze(-1).expand(-1, -1, self.num_heads),
            attn_contribution
        )

        attn = F.softmax(attn, dim=1)  # Normalize attention weights

        # 🔹 Apply Attention Weights to Values
        out_F = torch.zeros((B, M, self.num_heads, self.attn_channels), device=device, dtype=dtype)
        v_gathered = v[input_indices]

        weighted_v = attn[:, output_indices, :].unsqueeze(-1) * v_gathered  

        out_F.scatter_add_(
            1,
            output_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            .expand(B, -1, self.num_heads, self.attn_channels),
            weighted_v
        )

        # 🔹 Output Projection & Global Feature Aggregation
        out_projected = self.to_out(out_F.view(B, M, -1))  
        out_permuted = out_projected.permute(0, 2, 1)  
        out = self.max_pool(out_permuted).squeeze(-1)  

        return out, attn
