import torch.nn as nn
from torchvision import models
from src.models.base import BaseModel


class MobileNetV2(BaseModel):
    """
    MobileNetV2 fine-tuned for binary chest X-ray classification.

    Uses ImageNet pretrained weights. The classifier head is replaced to
    output num_classes logits. All layers are unfrozen for full fine-tuning.

    MobileNetV2 is included as the efficiency anchor in this benchmark —
    it was designed for mobile and edge deployment with minimal compute.

    Parameter count: ~3.4M
    Reference: Sandler et al., "MobileNetV2: Inverted Residuals and
               Linear Bottlenecks", 2018.

    Args:
        num_classes: Number of output classes. Default: 2.
        pretrained: Load ImageNet pretrained weights. Default: True.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.network = models.mobilenet_v2(weights=weights)

        # Replace classifier head — MobileNetV2 classifier is a Sequential
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.network(x)

    @property
    def model_name(self) -> str:
        return "mobilenetv2"