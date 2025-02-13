import torch
import torch.nn.functional as F
import torch.nn as nn

from attention_vector.point_cloud_attn_vector.modules.local_base_self_attention import LocalSelfAttentionBase
from attention_vector.point_cloud_attn_vector.utils.sebottleneck import SEBottleneck

class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    """
    A lightweight self-attention layer for sparse point cloud data, operating in 3D space.
    This layer processes sparse tensors efficiently by leveraging kernel-based neighborhood computations.
    """

    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 3, num_heads: int = 4, sparse_ratio: float = 0.5):
        """
        Initializes the LightweightSelfAttentionLayer.
        """
        super().__init__(kernel_size=kernel_size, dimension=3, sparse_ratio=sparse_ratio)

        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        # Squeeze-and-Excitation Bottleneck before Attention
        self.feature_bottleneck = SEBottleneck(in_channels)

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
        """
        sparse_x = sparse_x.coalesce()
        sparse_values = sparse_x.values()

        # Apply Squeeze-and-Excitation Bottleneck (Adaptive Feature Selection)
        sparse_values = self.feature_bottleneck(sparse_values.unsqueeze(-1)).squeeze(-1)

        # Compute Queries, Keys, and Values
        q = self.to_query(sparse_values).view(-1, self.num_heads, self.attn_channels)
        v = self.to_value(sparse_values).view(-1, self.num_heads, self.attn_channels)

        # Kernel Mapping for Neighborhood Indices
        kernel_map, out_coordinates = self.get_kernel_map_and_out_key(sparse_x)

        # Prepare Tensors for Sparse Attention Computation
        input_indices = kernel_map[:, 0]
        output_indices = kernel_map[:, 1]
        rel_pos_indices = kernel_map[:, 2]

        # Compute Attention Contribution
        attn_contribution = (F.normalize(q[input_indices], dim=-1) *
                             F.normalize(self.inter_pos_enc[rel_pos_indices].unsqueeze(0), dim=-1)).sum(dim=-1)

        # Compute Sparse Attention Weights
        attn = torch.zeros((sparse_x.shape[0], out_coordinates.shape[0], self.num_heads), device=sparse_x.device, dtype=sparse_x.dtype)

        # DEBUGGING
        #----------------------#
        # Print BEFORE fixing shapes
        # print(f"b_attn.shape: {attn.shape}")  # Expected: (batch_size, num_voxels, num_heads)
        # print(f"b_output_indices.shape BEFORE fix: {output_indices.shape}")  # Expected: (batch_size, num_voxels)
        # print(f"b_attn_contribution.shape: {attn_contribution.shape}")  # Expected: (batch_size, num_voxels, num_heads)
        #----------------------#


        output_indices = output_indices.view(1, -1).expand(attn.shape[0], -1)
        output_indices = output_indices.squeeze(-1)
        output_indices = output_indices.unsqueeze(-1).expand(-1, -1, attn.shape[2])

        # DEBUGGING
        #----------------------#
        # Print AFTER fixing shapes
        # print(f"a_attn.shape: {attn.shape}")  # Should be (batch_size, num_voxels, num_heads)
        # print(f"a_output_indices.shape: {output_indices.shape}")  # Should match attn[:2]
        # print(f"a_attn_contribution.shape: {attn_contribution.shape}")  # Should match attn
        #----------------------#

        attn_contribution = attn_contribution.expand(attn.shape)
        attn.scatter_add_(1, output_indices, attn_contribution)

        # Apply Attention Weights to Values
        weighted_v = attn.gather(1, output_indices).unsqueeze(-1) * v[input_indices].unsqueeze(0).expand(attn.shape[0], -1, -1, -1)
        out_F = torch.zeros((sparse_x.shape[0], out_coordinates.shape[0], self.num_heads, self.attn_channels), device=sparse_x.device, dtype=sparse_x.dtype)
        out_F.scatter_add_(1, output_indices.unsqueeze(-1).expand(-1, -1, self.num_heads, self.attn_channels), weighted_v)

        # Output Projection & Global Feature Aggregation
        out_projected = self.to_out(out_F.view(sparse_x.shape[0], -1, self.out_channels))
        out_permuted = out_projected.permute(0, 2, 1)
        out = self.max_pool(out_permuted).squeeze(-1)

        return out, attn