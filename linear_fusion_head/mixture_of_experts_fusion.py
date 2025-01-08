import torch
import torch.nn as nn
from base_fusion_head import BaseFusionHead

class MoEFusionHead(BaseFusionHead):
    def __init__(self, feature_dim, output_dim=1):
        """
        Mixture of Experts (MoE) fusion head.

        Args:
            feature_dim (int): Dimensionality of input features.
            output_dim (int): Output dimensionality (default: 1 for binary classification).
        """
        super(MoEFusionHead, self).__init__(feature_dim)
        self.expert_1 = nn.Linear(feature_dim, 64)
        self.expert_2 = nn.Linear(feature_dim, 64)
        self.expert_3 = nn.Linear(feature_dim, 64)
        self.gate = nn.Linear(feature_dim * 3, 3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        """
        Forward pass for MoE fusion.

        Args:
            inputs: List of input tensors to process through experts.

        Returns:
            torch.Tensor: Output of the fusion head.
        """
        # Experts
        expert_1_out = self.expert_1(self.flatten(inputs[0]))
        expert_2_out = self.expert_2(self.flatten(inputs[1]))
        expert_3_out = self.expert_3(self.flatten(inputs[2]))
        
        # Gating mechanism
        concatenated = torch.cat([expert_1_out, expert_2_out, expert_3_out], dim=1)
        gates = torch.softmax(self.gate(concatenated), dim=1)
        weighted_output = (gates[:, 0] * expert_1_out +
                           gates[:, 1] * expert_2_out +
                           gates[:, 2] * expert_3_out)
        
        # Final layers
        x = self.fc1(weighted_output)
        return self.sigmoid(self.fc2(x))
