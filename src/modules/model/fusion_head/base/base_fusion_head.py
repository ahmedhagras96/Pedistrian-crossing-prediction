from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFusionHead(nn.Module, ABC):
    """
    Base class for fusion heads. 
    All custom fusion classes should inherit from this class.
    """

    def __init__(self, feature_dim: int):
        """
        Initialize the BaseFusionHead.

        Args:
            feature_dim (int): Dimensionality of the input features.
        """
        super().__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion head. 
        This method must be implemented by all derived classes.

        Args:
            *inputs (torch.Tensor): A variable number of input tensors for fusion.

        Returns:
            torch.Tensor: The output tensor after fusion.
        """
        pass

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flattens the input tensor into shape (batch_size, -1).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, -1).
        """
        return x.view(x.size(0), -1)
