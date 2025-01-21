import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention.pointcloud_attention.layers.local_self_attention_base import LocalSelfAttentionBase
from modules.config.logger import Logger


class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    """
    A lightweight self-attention layer for point cloud data, operating in 3D space.
    Inherits from LocalSelfAttentionBase for kernel-based neighborhood computations.
    It applies both absolute (intra) and relative (inter) positional encodings, then
    performs multi-head attention using a cosine-similarity-based approach.
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_heads=4):
        """
        Initialize the LightweightSelfAttentionLayer.

        Args:
            in_channels (int):
                Number of input feature channels per point.
            out_channels (int, optional):
                Number of output feature channels after attention. If None,
                set to in_channels. Must be divisible by num_heads.
            kernel_size (int or tuple, optional):
                Size of the kernel used in neighborhood mapping. Defaults to 3.
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

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        # Linear layers to project input features to query/value and then back out
        self.to_query = nn.Linear(in_channels, out_channels)
        self.to_value = nn.Linear(in_channels, out_channels)
        self.to_out = nn.Linear(out_channels, out_channels)

        # Absolute (intra) positional encoding
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

        # Relative (inter) positional encoding
        self.inter_pos_enc = nn.Parameter(
            torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels)
        )
        nn.init.normal_(self.inter_pos_enc, mean=0.0, std=1.0)

        # Max pooling for global aggregation
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # self.logger.debug(f"{self.__class__.__name__} successfully initialized.")

    def forward(self, x: torch.Tensor, norm_points: torch.Tensor, batch_indices: torch.Tensor):
        """
        Forward pass for lightweight self-attention on point cloud data.

        Args:
            x (torch.Tensor):
                Input feature tensor of shape [B, N, C], where B is batch size,
                N is number of points, and C is number of input channels.
            norm_points (torch.Tensor):
                Normalized point coordinates of shape [B, N, 3]. Used for both
                absolute and relative positional encodings.
            batch_indices (torch.Tensor):
                Tensor of shape [B, N] (or possibly flattened [B*N]) indicating the
                batch index for each point.

        Returns:
            out (torch.Tensor):
                Aggregated output of shape [B, out_channels], after pooling.
            attn (torch.Tensor):
                Attention map of shape [B, M, num_heads], where M is the number of
                unique output coordinates from the kernel mapping process.
        """
        B, N, C = x.shape
        # self.logger.debug(f"Forward called with x of shape {x.shape}, "
        #                   f"norm_points of shape {norm_points.shape}, "
        #                   f"batch_indices of shape {batch_indices.shape}.")

        # 1) Absolute positional encoding (intra)
        # ---------------------------------------
        # Flatten points [B*N, 3], apply MLP, reshape back to [B, N, C]
        intra_pos_enc = self.intra_pos_mlp(norm_points.view(-1, 3)).view(B, N, C)
        x = x + intra_pos_enc  # Residual addition

        # 2) Compute queries (q) and values (v)
        # -------------------------------------
        # Reshape for multi-head attention
        q = self.to_query(x).view(B, N, self.num_heads, self.attn_channels)
        v = self.to_value(x).view(B, N, self.num_heads, self.attn_channels)

        # 3) Kernel mapping for neighborhood indices
        # ------------------------------------------
        # Coordinates should be integer for offset-based computation
        coordinates = norm_points.long()
        kernel_map, out_coordinates = self.get_kernel_map_and_out_key(coordinates, batch_indices)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)
        M = out_coordinates.shape[0]

        device = x.device
        dtype = x.dtype

        # 4) Prepare tensors for attention scatter operations
        # ---------------------------------------------------
        # input_indices: [K]
        input_indices = torch.tensor([kq[0] for kq in kq_map], device=device, dtype=torch.long)
        # output_indices: [K]
        output_indices = torch.tensor([kq[1] for kq in kq_map], device=device, dtype=torch.long)
        # rel_pos_indices: [K]
        rel_pos_indices = torch.tensor([kq[2] for kq in kq_map], device=device, dtype=torch.long)

        # 5) Normalize q and relative position encodings for cosine similarity
        # -------------------------------------------------------------------
        norm_q = F.normalize(q, p=2, dim=-1)  # [B, N, num_heads, attn_channels]
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)  # [kernel_volume, num_heads, attn_channels]

        # Gather per-neighbor q and inter_pos_enc
        q_gathered = norm_q[:, input_indices, :, :]  # [B, K, num_heads, attn_channels]
        pos_enc_gathered = norm_pos_enc[rel_pos_indices, :, :]  # [K, num_heads, attn_channels]

        # 6) Compute attention contributions via cosine similarity
        # ---------------------------------------------------------
        # (q_gathered * pos_enc) -> sum over last dim
        attn_contribution = (q_gathered * pos_enc_gathered.unsqueeze(0)).sum(dim=-1)  # [B, K, num_heads]

        # Scatter-add attention contributions into [B, M, num_heads]
        attn = torch.zeros((B, M, self.num_heads), device=device, dtype=dtype)
        attn.scatter_add_(
            1,
            output_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.num_heads),
            attn_contribution
        )

        # Normalize to get final attention coefficients
        attn = F.softmax(attn, dim=1)  # [B, M, num_heads]

        # 7) Use attention to weigh values (v)
        # -------------------------------------
        out_F = torch.zeros((B, M, self.num_heads, self.attn_channels), device=device, dtype=dtype)
        v_gathered = v[:, input_indices, :, :]  # [B, K, num_heads, attn_channels]

        # Weighted values based on attention
        weighted_v = attn[:, output_indices, :].unsqueeze(-1) * v_gathered  # [B, K, num_heads, attn_channels]

        # Scatter-add weighted values into out_F
        out_F.scatter_add_(
            1,
            output_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            .expand(B, -1, self.num_heads, self.attn_channels),
            weighted_v
        )

        # 8) Output projection + global max pooling
        # -----------------------------------------
        # Combine multi-head features, project to out_channels
        out_projected = self.to_out(out_F.view(B, M, -1))  # [B, M, out_channels]

        # Pool across spatial dimension M
        out_permuted = out_projected.permute(0, 2, 1)  # [B, out_channels, M]
        out = self.max_pool(out_permuted).squeeze(-1)  # [B, out_channels]

        return out, attn
