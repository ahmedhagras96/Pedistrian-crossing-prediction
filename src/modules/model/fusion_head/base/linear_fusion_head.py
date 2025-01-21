import torch
import torch.nn as nn

from modules.model.fusion_head.base.base_fusion_head import BaseFusionHead


class LinearFusionHead(BaseFusionHead):
    """
    Linear fusion head that applies separate linear transformations to multiple feature 
    inputs (3D, pedestrian, behavior), concatenates them, and then applies a final 
    linear layer for binary classification.
    """

    def __init__(
            self,
            feature_dim_3d: int,
            feature_dim_ped: int,
            feature_dim_beh: int,
            fused_dim: int
    ):
        """
        Initialize the LinearFusionHead.

        Args:
            feature_dim_3d (int): Dimensionality of the 3D feature input.
            feature_dim_ped (int): Dimensionality of the pedestrian feature input.
            feature_dim_beh (int): Dimensionality of the behavior feature input.
            fused_dim (int): Dimensionality to which each feature input is projected 
                before concatenation.
        """
        super().__init__(feature_dim=None)  # You may adjust feature_dim as needed

        # Linear layers to normalize each input's dimensionality
        self.linear_3d = nn.Linear(feature_dim_3d, fused_dim)
        self.linear_ped = nn.Linear(feature_dim_ped, fused_dim)
        self.linear_beh = nn.Linear(feature_dim_beh, fused_dim)

        # Final fusion layer
        # By default, this layer outputs a single logit for binary classification.
        self.fusion_layer = nn.Linear(fused_dim * 3, 1)

    def forward(
            self,
            feat_3d: torch.Tensor,
            feat_ped: torch.Tensor,
            feat_beh: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the linear fusion head.

        Args:
            feat_3d (torch.Tensor): 3D feature input of shape (batch_size, feature_dim_3d).
            feat_ped (torch.Tensor): Pedestrian feature input of shape (batch_size, feature_dim_ped).
            feat_beh (torch.Tensor): Behavior feature input of shape (batch_size, feature_dim_beh).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), containing the 
                binary classification logits (after sigmoid).
        """
        # Normalize and concatenate features
        fused_features = self._normalize_and_concatenate(
            feat_3d, feat_ped, feat_beh
        )

        # Final classification output (logits converted to probabilities via sigmoid)
        output = self.fusion_layer(fused_features)
        return torch.sigmoid(output)

    def _normalize_and_concatenate(
            self,
            feat_3d: torch.Tensor,
            feat_ped: torch.Tensor,
            feat_beh: torch.Tensor
    ) -> torch.Tensor:
        """
        Private helper method to apply linear transformations to inputs and concatenate them.

        Args:
            feat_3d (torch.Tensor): 3D feature input of shape (batch_size, feature_dim_3d).
            feat_ped (torch.Tensor): Pedestrian feature input of shape (batch_size, feature_dim_ped).
            feat_beh (torch.Tensor): Behavior feature input of shape (batch_size, feature_dim_beh).

        Returns:
            torch.Tensor: Concatenated feature tensor of shape (batch_size, fused_dim * 3).
        """
        norm_3d = self.linear_3d(feat_3d)  # [batch_size, fused_dim]
        norm_ped = self.linear_ped(feat_ped)  # [batch_size, fused_dim]
        norm_beh = self.linear_beh(feat_beh)  # [batch_size, fused_dim]

        # Concatenate features along the last dimension
        return torch.cat([norm_3d, norm_ped, norm_beh], dim=1)
