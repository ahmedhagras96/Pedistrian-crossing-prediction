import torch
import torch.nn as nn

from modules.model.fusion_head.base.base_fusion_head import BaseFusionHead


class AttentionFusionHead(BaseFusionHead):
    """
    End-to-end neural network with attention-based fusion. 
    Concatenates multiple input sequences and processes them with 
    a multi-head attention mechanism, followed by a small feedforward network.
    """

    def __init__(self, vector_dim: int = 64, num_heads: int = 4):
        """
        Initialize the AttentionFusionHead.

        Args:
            vector_dim (int, optional): Dimensionality of attention vectors from each source. 
                Defaults to 64.
            num_heads (int, optional): Number of attention heads for the multi-head attention. 
                Defaults to 4.
        """
        super().__init__(feature_dim=3)  # '3' might be arbitrary, depending on your base class usage

        # Multi-head attention fusion layer.
        # Note: 'batch_first=True' allows input/output shapes (batch_size, seq_len, embed_dim).
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=vector_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Fully connected layers after attention-based fusion
        self.fc1 = nn.Linear(vector_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            source1: torch.Tensor,
            source2: torch.Tensor,
            source3: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention-based fusion network.

        Args:
            source1 (torch.Tensor): Attention vectors from the first source, 
                of shape (batch_size, seq_len, vector_dim).
            source2 (torch.Tensor): Attention vectors from the second source, 
                of shape (batch_size, seq_len, vector_dim).
            source3 (torch.Tensor): Attention vectors from the third source, 
                of shape (batch_size, seq_len, vector_dim).

        Returns:
            torch.Tensor: Binary classification output of shape (batch_size, 1).
        """
        concatenated_inputs = self._concatenate_inputs(source1, source2, source3)
        fusion_output = self._apply_attention(concatenated_inputs)
        pooled_output = self._pool_sequence(fusion_output)
        output = self._apply_classifier(pooled_output)
        return output

    def _concatenate_inputs(
            self,
            source1: torch.Tensor,
            source2: torch.Tensor,
            source3: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate three input tensors along the sequence dimension.

        Args:
            source1 (torch.Tensor): First input sequence.
            source2 (torch.Tensor): Second input sequence.
            source3 (torch.Tensor): Third input sequence.

        Returns:
            torch.Tensor: Concatenated tensor of shape 
                (batch_size, seq_len * 3, vector_dim).
        """
        return torch.cat([source1, source2, source3], dim=1)

    def _apply_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to the concatenated input sequences.

        Args:
            inputs (torch.Tensor): Concatenated sequence of shape 
                (batch_size, total_seq_len, embed_dim).

        Returns:
            torch.Tensor: Output after multi-head attention of the same shape 
                (batch_size, total_seq_len, embed_dim).
        """
        # In multi-head attention, the same tensor is used for query, key, and value here.
        fusion_output, _ = self.fusion_attention(inputs, inputs, inputs)
        return fusion_output

    def _pool_sequence(self, fusion_output: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling across the sequence dimension.

        Args:
            fusion_output (torch.Tensor): Attention outputs of shape 
                (batch_size, total_seq_len, embed_dim).

        Returns:
            torch.Tensor: Pooled tensor of shape (batch_size, embed_dim).
        """
        return torch.mean(fusion_output, dim=1)

    def _apply_classifier(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Pass the pooled output through fully connected layers and apply sigmoid.

        Args:
            pooled_output (torch.Tensor): Pooled attention output of shape 
                (batch_size, embed_dim).

        Returns:
            torch.Tensor: Final classification output of shape (batch_size, 1).
        """
        x = self.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
