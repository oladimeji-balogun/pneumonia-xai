import torch.nn as nn
from torchvision import models
from src.models.base import BaseModel


class ResNet18(BaseModel):
    """
    ResNet-18 fine-tuned for binary chest X-ray classification.

    Uses ImageNet pretrained weights. The final fully connected layer is
    replaced to output num_classes logits. All layers are unfrozen for
    full fine-tuning.

    Parameter count: ~11.2M
    Reference: He et al., "Deep Residual Learning for Image Recognition", 2015.

    Args:
        num_classes: Number of output classes. Default: 2.
        pretrained: Load ImageNet pretrained weights. Default: True.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.network = models.resnet18(weights=weights)

        # Replace classifier head
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)

    @property
    def model_name(self) -> str:
        return "resnet18"