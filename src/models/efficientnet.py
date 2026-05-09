import torch.nn as nn
from torchvision import models
from src.models.base import BaseModel


class EfficientNetB0(BaseModel):
    """
    EfficientNet-B0 fine-tuned for binary chest X-ray classification.

    Uses ImageNet pretrained weights. The classifier head is replaced to
    output num_classes logits. All layers are unfrozen for full fine-tuning.

    EfficientNet-B0 sits on the accuracy-efficiency Pareto frontier in this
    benchmark — it achieves strong accuracy at moderate computational cost
    through compound scaling of depth, width, and resolution.

    Parameter count: ~5.3M
    Reference: Tan & Le, "EfficientNet: Rethinking Model Scaling for
               Convolutional Neural Networks", 2019.

    Args:
        num_classes: Number of output classes. Default: 2.
        pretrained: Load ImageNet pretrained weights. Default: True.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.network = models.efficientnet_b0(weights=weights)

        # Replace classifier head — EfficientNet classifier is a Sequential
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.network(x)

    @property
    def model_name(self) -> str:
        return "efficientnet_b0"