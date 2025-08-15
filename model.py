import torch
import torch.nn as nn

from attention_vector.Pedestrian_PC_Attention.Ped_Att_Model import PointNetFeatureExtractor
from attention_vector.point_cloud_attn_vector.attention_model import PointCloudAttentionModel
from features_extraction.Multi_head_attention_layer import MultiHeadAttention
from linear_fusion_head.attention_Base_fusion import AttentionFusionHead

class PedestrianCrossingFusionNet(nn.Module):
    """
    A neural network that fuses pedestrian point cloud data, environment data, and features
    using multi-head attention and a lightweight self-attention mechanism.
    """

    def __init__(self, embed_dim=64, num_heads=4, kernel_size=3, max_voxel_grid_size=int(1e5), sparse_ratio=0.5, voxel_size=1.0, feature_dim=5, feature_num_heads=5, dropout_rate=0.3):
        super(PedestrianCrossingFusionNet, self).__init__()
        
        # Feature extractor for pedestrian point clouds
        self.pointnet_feature_extractor = PointNetFeatureExtractor(output_dim=embed_dim)

        # Attention model for point cloud data
        self.point_cloud_attention_model = PointCloudAttentionModel(
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
            max_voxel_grid_size=max_voxel_grid_size,
            sparse_ratio=sparse_ratio,
            voxel_size=voxel_size
        )
        
        # Multi-head attention layer for feature fusion
        self.multi_head_attention = MultiHeadAttention(input_dim=feature_dim, num_heads=feature_num_heads, output_dim=embed_dim)
        
        # Fusion head for final classification
        self.fusion_head = AttentionFusionHead(vector_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)

    def forward(self, input_data):
        
        _, environment_pc, _ = input_data

        # Extract features from pedestrian point clouds
        # pedestrian_features = self.pointnet_feature_extractor(pedestrian_pc)
        
        # Apply attention to pedestrian point clouds
        environment_attention_output = self.point_cloud_attention_model(environment_pc)[0]
        
        # Apply multi-head attention to fuse pedestrian and environment features
        # fused_features = self.multi_head_attention(features)[0]
        
        # Final fusion head for classification
        output = self.fusion_head(environment_attention_output)

        return output
    

class LRWrapper(nn.Module):
    """
    A wrapper for the PedestrianCrossingFusionNet to handle input and output for LR finder.
    """
    
    def __init__(self, model):
        super(LRWrapper, self).__init__()
        self.model = model

    def forward(self, input_data):
        # input_data: (avatar, env, feats, target)
        avatar, env, feats = input_data
        # print("LRWrapper shapes:", avatar.shape, env.shape, feats.shape)
        return self.model((avatar, env, feats)).squeeze(-1)