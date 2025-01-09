import torch.nn as nn

from modules.attention_vector.point_cloud_attention.modules.light_weight_self_attention import LightweightSelfAttentionLayer
from modules.attention_vector.point_cloud_attention.modules.centroid_aware_voxelization import CentroidAwareVoxelization


class PointCloudAttentionModel(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, stride=1, dilation=1, num_heads=4):
        super(PointCloudAttentionModel, self).__init__()
        self.CAV = CentroidAwareVoxelization(embed_dim=embed_dim)
        self.attn_layer = LightweightSelfAttentionLayer(
            in_channels=embed_dim*2,
            out_channels=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )

    def forward(self, x):
        aggregated_features, norm_points, voxel_centroids, voxel_counts, pos_embs, batch_ids = self.CAV(x)
        out, wei = self.attn_layer(aggregated_features, norm_points, batch_ids)

        return out, wei