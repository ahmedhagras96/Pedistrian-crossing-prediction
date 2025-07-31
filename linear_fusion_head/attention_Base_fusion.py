import torch
import torch.nn as nn

class AttentionFusionHead(nn.Module):
    def __init__(self, vector_dim, num_heads=4, dropout_rate=0.3):
        """
        Neural network with attention-based fusion, but without sequence modeling.

        Args:
            vector_dim (int): Dimensionality of attention vectors from each source.
            num_heads (int): Number of attention heads for fusion.
        """
        super(AttentionFusionHead, self).__init__()

        # More fully connected layers for increased complexity
        input_dim = 3 * vector_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.fc6 = nn.Linear(32, 1)

        # Activation and dropout
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)

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
        x = torch.cat([source1, source2, source3], dim=1)  # (batch_size, 3 * vector_dim)

        # More fully connected layers
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 3 with residual connection
        residual = x
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + residual

        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 5
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.activation(x)

        # Output layer (raw logits for BCEWithLogitsLoss)
        output = self.fc6(x)

        return output
