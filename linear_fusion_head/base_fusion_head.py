import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseFusionHead(nn.Module, ABC):
    """
    Base class for fusion heads.
    All fusion classes should inherit from this class.
    """
    def __init__(self, feature_dim):
        """
        Initialize the BaseFusionHead.

        Args:
            feature_dim (int): Dimensionality of input features.
        """
        super(BaseFusionHead, self).__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, *inputs):
        """
        Abstract method for the forward pass. 
        Must be implemented by all derived classes.

        Args:
            inputs: Variable number of inputs for fusion.
        
        Returns:
            torch.Tensor: Output of the fusion head.
        """
        pass

    def flatten(self, x):
        """
        Utility function to flatten input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, -1).
        """
        return x.view(x.size(0), -1)
