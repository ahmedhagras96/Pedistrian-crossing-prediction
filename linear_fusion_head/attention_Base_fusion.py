# import torch
# import torch.nn as nn
# from .base_fusion_head import BaseFusionHead

# class AttentionFusionHead(BaseFusionHead):
#     def __init__(self,batch_size, vector_dim, num_heads=4):
#         """
#         End-to-end neural network with attention-based fusion.

#         Args:
#             vector_dim (int): Dimensionality of attention vectors from each source (default: 64).
#             num_heads (int): Number of attention heads for fusion.
#         """
#         super(AttentionFusionHead, self).__init__(feature_dim=3)

#         # Attention-based fusion layer
#         self.fusion_attention = nn.MultiheadAttention(embed_dim=vector_dim, num_heads=num_heads)

#         # Fully connected layers after fusion
#         self.fc1 = nn.Linear(vector_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)

#         # Activation and dropout
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, source1, source2, source3):
#         """
#         Forward pass of the network.

#         Args:
#             source1 (torch.Tensor): Attention vectors from 3D constructed data (batch_size, seq_len, vector_dim).
#             source2 (torch.Tensor): Attention vectors from 3D pedestrian point cloud (batch_size, seq_len, vector_dim).
#             source3 (torch.Tensor): Attention vectors from pedestrian features (batch_size, seq_len, vector_dim).

#         Returns:
#             torch.Tensor: Binary classification output (batch_size, 1).
#         """
#         # Concatenate inputs along the sequence dimension
#         concatenated_inputs = torch.cat([source1, source2, source3], dim=1)  # (batch_size, 3 * seq_len, vector_dim)
#         print("concatenated_inputs: ",concatenated_inputs.shape)

#         # Apply attention-based fusion
#         fusion_output, _ = self.fusion_attention(concatenated_inputs, concatenated_inputs, concatenated_inputs)  # (batch_size, 3 * seq_len, vector_dim)
#         print("fusion_output: ",fusion_output.shape)

#         # Average pooling across the sequence length
#         pooled_output = torch.mean(fusion_output, dim=1)  # (batch_size, vector_dim)
#         print("pooled_output: ",pooled_output.shape)

#         # Fully connected layers
#         x = self.relu(self.fc1(pooled_output))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         output = self.sigmoid(self.fc3(x))  # (batch_size, 1)

#         return output
import torch
import torch.nn as nn

class AttentionFusionHead(nn.Module):
    def __init__(self, vector_dim, num_heads=4):
        """
        Neural network with attention-based fusion, but without sequence modeling.

        Args:
            vector_dim (int): Dimensionality of attention vectors from each source.
            num_heads (int): Number of attention heads for fusion.
        """
        super(AttentionFusionHead, self).__init__()

        # More fully connected layers for increased complexity
        self.fc1 = nn.Linear(3 * vector_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, source1, source2, source3):
        """
        Forward pass of the network.

        Args:
            source1 (torch.Tensor): Attention vectors from 3D constructed data (batch_size, vector_dim).
            source2 (torch.Tensor): Attention vectors from 3D pedestrian point cloud (batch_size, vector_dim).
            source3 (torch.Tensor): Attention vectors from pedestrian features (batch_size, vector_dim).

        Returns:
            torch.Tensor: Binary classification output (batch_size, 1).
        """
        # Concatenate inputs along the feature dimension
        concatenated_inputs = torch.cat([source1, source2, source3], dim=1)  # (batch_size, 3 * vector_dim)

        # More fully connected layers
        x = self.relu(self.fc1(concatenated_inputs))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        output = self.fc6(x)  # (batch_size, 1)

        return output
