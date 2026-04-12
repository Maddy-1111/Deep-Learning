import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        fine_tune: str = 'full',
    ):
        super().__init__()
        self.encoder = VGG11Encoder(
            in_channels=in_channels,
            fine_tune=fine_tune,
            use_batchnorm=use_batchnorm,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._make_head(
            classes=num_classes,
            drop_prob=dropout_p,
            use_batchnorm=use_batchnorm,
        )

    def _make_head(self, classes: int, drop_prob: float, use_batchnorm: bool) -> nn.Sequential:
        # Two-layer MLP head that maps pooled VGG features to class logits.
        norm = (lambda width: nn.BatchNorm1d(width)) if use_batchnorm else (lambda _: nn.Identity())
        width = 4096
        input_dim = 512 * 7 * 7

        layers = [
            nn.Linear(input_dim, width),
            norm(width),
            nn.ReLU(inplace=True),
            CustomDropout(p=drop_prob),
            nn.Linear(width, width),
            norm(width),
            nn.ReLU(inplace=True),
            CustomDropout(p=drop_prob),
            nn.Linear(width, classes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode image -> pool -> flatten -> per-class scores.
        encoded = self.encoder(x)
        pooled = self.avgpool(encoded)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)