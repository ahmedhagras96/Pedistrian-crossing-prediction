# Example PyTorch code snippet
import torch
import torch.nn as nn

class LinearFusionHead(nn.Module):
    def __init__(self, feature_dim_3d, feature_dim_ped, feature_dim_beh, fused_dim):
        super(LinearFusionHead, self).__init__()
        # Linear layers to normalize feature dimensions
        self.linear_3d = nn.Linear(feature_dim_3d, fused_dim)
        self.linear_ped = nn.Linear(feature_dim_ped, fused_dim)
        self.linear_beh = nn.Linear(feature_dim_beh, fused_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(fused_dim * 3, 1)  # Outputs a binary classification
    
    def forward(self, feat_3d, feat_ped, feat_beh):
        # Normalize dimensions
        norm_3d = self.linear_3d(feat_3d)  # [B, fused_dim]
        norm_ped = self.linear_ped(feat_ped)  # [B, fused_dim]
        norm_beh = self.linear_beh(feat_beh)  # [B, fused_dim]
        
        # Concatenate features
        fused_features = torch.cat([norm_3d, norm_ped, norm_beh], dim=1)  # [B, fused_dim * 3]
        
        # Fusion and classification
        output = self.fusion_layer(fused_features)  # [B, 1]
        return torch.sigmoid(output)  # Binary classification probability
