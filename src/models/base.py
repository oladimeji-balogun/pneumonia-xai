from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models in the benchmarking suite.

    All architectures — pretrained wrappers and the custom CNN — must
    inherit from this class. This ensures the trainer, evaluator, and
    benchmark modules can operate on any model through a uniform interface
    without caring about the underlying architecture.

    Subclasses must implement:
        - forward(x): standard PyTorch forward pass
        - model_name (property): unique string identifier for the architecture
        - param_count (property): total number of trainable parameters
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique string identifier for this architecture (e.g. 'resnet18')."""
        pass

    @property
    def param_count(self) -> Dict[str, int]:
        """
        Returns total and trainable parameter counts.

        Returns:
            Dict with keys 'total' and 'trainable'.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}