import torch
import torch.nn as nn
from src.models.base import BaseModel


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.

    Factorizes a standard convolution into a depthwise convolution (one filter
    per input channel) followed by a pointwise convolution (1x1). This reduces
    parameter count and FLOPs significantly compared to standard convolutions
    while preserving representational capacity.

    Used in MobileNet family — we adopt it here as the core building block
    of our lightweight architecture.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for depthwise conv. Default: 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise: one filter per input channel
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise: mix channels with 1x1 conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCXR(BaseModel):
    """
    LightCXR: A lightweight CNN for chest X-ray pneumonia detection.

    Architecture designed explicitly for resource-constrained deployment:
    - Depthwise separable convolutions as primary building blocks
    - Progressive channel expansion with aggressive spatial downsampling
    - Global average pooling instead of fully connected layers to minimize
      parameters at the classification head
    - Target: < 500K trainable parameters

    Architecture overview:
        Stem conv (3 -> 32)
        DS Block (32 -> 64,  stride=2)  -> 112x112
        DS Block (64 -> 128, stride=2)  -> 56x56
        DS Block (128 -> 128)           -> 56x56
        DS Block (128 -> 256, stride=2) -> 28x28
        DS Block (256 -> 256)           -> 28x28
        DS Block (256 -> 512, stride=2) -> 14x14
        Global Average Pooling          -> 512
        Classifier (512 -> num_classes)

    Args:
        num_classes: Number of output classes. Default: 2.
        dropout_rate: Dropout rate before classifier. Default: 0.3.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()

        # Stem: standard conv to extract low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Backbone: depthwise separable conv blocks
        self.backbone = nn.Sequential(
            DepthwiseSeparableConv(32,  64,  stride=2),
            DepthwiseSeparableConv(64,  128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
        )

        # Global average pooling: collapses spatial dims to 1x1
        # Eliminates large FC layers, dramatically reduces parameter count
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @property
    def model_name(self) -> str:
        return "lightcxr"