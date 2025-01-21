import torch
import torch.nn as nn

from modules.model.fusion_head.base.base_fusion_head import BaseFusionHead


class MoEFusionHead(BaseFusionHead):
    """
    Mixture of Experts (MoE) fusion head that processes multiple input tensors
    through separate expert networks and then uses a gating mechanism to
    combine their outputs.
    """

    def __init__(self, feature_dim: int, output_dim: int = 1):
        """
        Initialize the MoEFusionHead.

        Args:
            feature_dim (int): Dimensionality of the input features to each expert.
            output_dim (int): Output dimensionality. Defaults to 1 for binary classification.
        """
        super().__init__(feature_dim)

        # Expert layers
        self.expert_1 = nn.Linear(feature_dim, 64)
        self.expert_2 = nn.Linear(feature_dim, 64)
        self.expert_3 = nn.Linear(feature_dim, 64)

        # Gating layer (expects concatenated expert outputs)
        self.gate = nn.Linear(feature_dim * 3, 3)

        # Final classification layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE fusion.

        Args:
            *inputs (torch.Tensor): A variable number of input tensors. 
                For this implementation, exactly three inputs are expected.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim), 
                containing the final fusion result after gating and classification.
        """
        # Compute expert outputs
        expert_outputs = self._compute_expert_outputs(inputs)

        # Compute gating values using concatenated inputs
        gates = self._compute_gates(inputs)

        # Weighted combination of expert outputs
        weighted_output = self._apply_gating(expert_outputs, gates)

        # Final classification layers
        x = self.fc1(weighted_output)
        return self.sigmoid(self.fc2(x))

    def _compute_expert_outputs(self, inputs: tuple) -> list:
        """
        Compute outputs for each expert network.

        Args:
            inputs (tuple of torch.Tensor): The input tensors for each expert.
                Expected length is three in this example.

        Returns:
            list: A list of expert outputs. Each output is of shape (batch_size, 64).
        """
        expert_1_out = self.expert_1(self.flatten(inputs[0]))
        expert_2_out = self.expert_2(self.flatten(inputs[1]))
        expert_3_out = self.expert_3(self.flatten(inputs[2]))
        return [expert_1_out, expert_2_out, expert_3_out]

    def _compute_gates(self, inputs: tuple) -> torch.Tensor:
        """
        Compute gate values using a linear layer over the concatenated raw inputs.

        Args:
            inputs (tuple of torch.Tensor): The same input tensors for each expert.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 3) representing 
                the normalized gating coefficients for each expert.
        """
        # Concatenate raw (flattened) inputs for the gating layer
        concatenated = torch.cat([self.flatten(inp) for inp in inputs], dim=1)
        return torch.softmax(self.gate(concatenated), dim=1)

    def _apply_gating(self, expert_outputs: list, gates: torch.Tensor) -> torch.Tensor:
        """
        Combine expert outputs via the computed gating values.

        Args:
            expert_outputs (list): List of expert output tensors. 
                Each tensor is of shape (batch_size, 64).
            gates (torch.Tensor): Softmax-normalized weights for each expert of 
                shape (batch_size, 3).

        Returns:
            torch.Tensor: Weighted sum of expert outputs. 
                Shape is (batch_size, 64).
        """
        # gates[:, 0], gates[:, 1], gates[:, 2] select the gating coefficients
        return (
                gates[:, 0].unsqueeze(1) * expert_outputs[0] +
                gates[:, 1].unsqueeze(1) * expert_outputs[1] +
                gates[:, 2].unsqueeze(1) * expert_outputs[2]
        )
