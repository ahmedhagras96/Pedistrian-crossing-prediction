import torch.nn as nn

from attention_vector.point_cloud_attn_vector.modules import LightweightSelfAttentionLayer
from attention_vector.point_cloud_attn_vector.modules import CentroidAwareVoxelization


class PointCloudAttentionModel(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, num_heads=4):
        super(PointCloudAttentionModel, self).__init__()
        self.CAV = CentroidAwareVoxelization(embed_dim=embed_dim)
        self.attn_layer = LightweightSelfAttentionLayer(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            num_heads=num_heads, 
            kernel_size=kernel_size, 
            )
    
    def forward(self, x):
        aggregated_features, norm_points, voxel_centroids, voxel_counts, pos_embs, batch_ids = self.CAV(x)
        out, wei = self.attn_layer(aggregated_features, norm_points, batch_ids)
        
        return out, wei