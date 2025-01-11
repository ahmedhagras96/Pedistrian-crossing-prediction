import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention.point_cloud_attention.local_self_attention import LocalSelfAttention


class LightweightSelfAttentionLayer(LocalSelfAttention):
    """
    A lightweight self-attention layer designed for point cloud processing with local attention.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int, optional): Number of output feature channels. Defaults to `in_channels`.
        kernel_size (int): Size of the kernel for local attention.
        stride (int): Stride for kernel mapping.
        dilation (int): Dilation factor for kernel mapping.
        num_heads (int): Number of attention heads.

    Attributes:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        num_heads (int): Number of attention heads.
        attn_channels (int): Number of attention channels per heads.
        to_query (nn.Linear): Linear projection for query computation.
        to_value (nn.Linear): Linear projection for value computation.
        to_out (nn.Linear): Linear projection for output computation.
        intra_pos_mlp (nn.Sequential): MLP for computing absolute positional encoding.
        inter_pos_enc (nn.Parameter): Relative positional encoding for kernel offsets.
        max_pool (nn.AdaptiveMaxPool1d): Max pooling layer for global feature extraction.
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, dilation=1, num_heads=4):
        super(LightweightSelfAttentionLayer, self).__init__(kernel_size, stride, dilation, dimension=3)

        self.out_channels = out_channels or in_channels
        assert self.out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.attn_channels = self.out_channels // num_heads

        # Query, Value, and Output projections
        self.to_query = nn.Linear(in_channels, self.out_channels)
        self.to_value = nn.Linear(in_channels, self.out_channels)
        self.to_out = nn.Linear(self.out_channels, self.out_channels)

        # Absolute positional encoding
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )

        # Relative positional encoding
        self.inter_pos_enc = nn.Parameter(torch.randn(self.kernel_volume, self.num_heads, self.attn_channels))
        nn.init.normal_(self.inter_pos_enc, 0, 1)

        # Global feature extraction
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, features, norm_points, batch_indices):
        """
        Forward pass of the lightweight self-attention layer.

        Args:
            features (torch.Tensor): Input feature tensor of shape [B, N, in_channels].
            norm_points (torch.Tensor): Normalized point coordinates of shape [B, N, 3].
            batch_indices (torch.Tensor): Batch indices for each point of shape [B, N].

        Returns:
            tuple:
                - out (torch.Tensor): Aggregated output features of shape [B, out_channels].
                - attention (torch.Tensor): Attention map of shape [B, M, num_heads].
        """
        B, N, C = features.shape
        device = features.device

        # Apply absolute positional encoding
        abs_pos_enc = self.intra_pos_mlp(norm_points.view(-1, 3)).view(B, N, C)
        features = features + abs_pos_enc

        # Compute query and value tensors
        queries = self.to_query(features).view(B, N, self.num_heads, self.attn_channels)
        values = self.to_value(features).view(B, N, self.num_heads, self.attn_channels)

        # Kernel mapping for neighbors
        kernel_map, out_coords = self.compute_kernel_mapping(norm_points.long(), batch_indices)
        kq_map = self.map_key_query_indices(kernel_map)

        M = out_coords.shape[0]  # Number of output points

        # Initialize attention map
        attention = torch.zeros((B, M, self.num_heads), device=device)

        # Convert kernel mapping to tensors
        input_indices = torch.tensor([k[0] for k in kq_map], device=device, dtype=torch.long)
        output_indices = torch.tensor([k[1] for k in kq_map], device=device, dtype=torch.long)
        relative_indices = torch.tensor([k[2] for k in kq_map], device=device, dtype=torch.long)

        # Normalize queries and positional encodings
        norm_queries = F.normalize(queries, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)

        # Gather queries and relative positional encodings
        gathered_queries = norm_queries[:, input_indices, :, :]
        gathered_pos_enc = norm_pos_enc[relative_indices, :, :]

        # Compute attention contributions
        attn_contrib = (gathered_queries * gathered_pos_enc.unsqueeze(0)).sum(dim=-1)

        # Aggregate attention scores
        attention.scatter_add_(1, output_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.num_heads), attn_contrib)
        attention = F.softmax(attention, dim=1)

        # Initialize output features
        output_features = torch.zeros((B, M, self.num_heads, self.attn_channels), device=device)

        # Gather values and compute weighted values
        gathered_values = values[:, input_indices, :, :]
        weighted_values = attention[:, output_indices, :].unsqueeze(-1) * gathered_values

        # Aggregate weighted values
        output_features.scatter_add_(
            1,
            output_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.num_heads, self.attn_channels),
            weighted_values,
        )

        # Project output features
        out = self.to_out(output_features.view(B, M, -1))

        # Global feature extraction
        out = self.max_pool(out.permute(0, 2, 1)).squeeze(-1)

        return out, attention
