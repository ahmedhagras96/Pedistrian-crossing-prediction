import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.base import LocalSelfAttentionBase


class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, dilation=1, num_heads=4):
        super(LightweightSelfAttentionLayer, self).__init__(kernel_size, stride, dilation, dimension=3)

        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Linear(in_channels, out_channels)
        self.to_value = nn.Linear(in_channels, out_channels)
        self.to_out = nn.Linear(out_channels, out_channels)

        # Absolute positional encoding
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

        # Relative positional encoding
        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))
        nn.init.normal_(self.inter_pos_enc, 0, 1)
 
    def forward(self, x, norm_points, batch_indices):
        B, N, C = x.shape

        # Apply intra positional encoding
        intra_pos_enc = self.intra_pos_mlp(norm_points.view(-1, 3)).view(B, N, C)
        x = x + intra_pos_enc

        # Compute query and value tensors
        q = self.to_query(x).view(B, N, self.num_heads, self.attn_channels)  # [B, N, num_heads, attn_channels]
        v = self.to_value(x).view(B, N, self.num_heads, self.attn_channels)  # [B, N, num_heads, attn_channels]

        # Kernel mapping for neighbors
        coordinates = norm_points.long()  
        kernel_map, out_coordinates = self.get_kernel_map_and_out_key(coordinates, batch_indices)  # kernel_map: list, out_coordinates: [M, D]
        kq_map = self.key_query_map_from_kernel_map(kernel_map)  # List of tuples (input_index, output_index, rel_pos_idx)

        M = out_coordinates.shape[0]
        device = x.device
        dtype = x.dtype

        # Initialize 
        attn = torch.zeros((B, M, self.num_heads), device=device, dtype=dtype)  # [B, M, num_heads]

        # Convert kq_map to tensors for vectorization
        input_indices = torch.tensor([kq[0] for kq in kq_map], device=device, dtype=torch.long)  # [K]
        output_indices = torch.tensor([kq[1] for kq in kq_map], device=device, dtype=torch.long)  # [K]
        rel_pos_indices = torch.tensor([kq[2] for kq in kq_map], device=device, dtype=torch.long)  # [K]
        
        # Normalize query and positional encodings to get cosine similarity
        norm_q = F.normalize(q, p=2, dim=-1)  # [B, N, num_heads, attn_channels]
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)  # [kernel_volume, num_heads, attn_channels]
        # Gather all q and pos_enc
        q_gathered = norm_q[:, input_indices, :, :]  # [B, K, num_heads, attn_channels]
        pos_enc = norm_pos_enc[rel_pos_indices, :, :]  # [K, num_heads, attn_channels]

        # Compute attention contributions using cosine similarity instead of matrix multiplication
        attn_contribution = (q_gathered * pos_enc.unsqueeze(0)).sum(dim=-1)  # [B, K, num_heads]

        # Scatter add attention contributions
        attn.scatter_add_(1, output_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.num_heads), attn_contribution)

        attn = F.softmax(attn, dim=1)  # [B, M, num_heads]

        out_F = torch.zeros((B, M, self.num_heads, self.attn_channels), device=device, dtype=dtype)  # [B, M, num_heads, attn_channels]

        # Gather all v and compute weighted_v
        v_gathered = v[:, input_indices, :, :]  # [B, K, num_heads, attn_channels]
        weighted_v = attn[:, output_indices, :].unsqueeze(-1) * v_gathered  # [B, K, num_heads, attn_channels]

        # Scatter add weighted_v into out_F
        out_F.scatter_add_(1, output_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.num_heads, self.attn_channels), weighted_v)

        # Output projection
        wei = self.to_out(out_F.view(B, M, -1))  # [B, M, out_channels]
        out = torch.sum(wei, dim=1)  # [B, out_channels]
        return out, wei

