import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    def __init__(
        self,
        pretrained_path: str = None,
        fine_tune: str = 'full',
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            fine_tune=fine_tune,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.regression_head = self._build_regression_head(dropout_p)

    def _build_regression_head(self, dropout_p: float) -> nn.Sequential:
        # Predict normalized box coordinates (x1, y1, x2, y2) in [0, 1].
        hidden = 4096
        in_features = 512 * 7 * 7
        modules = [
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, image_size: torch.Tensor = None) -> torch.Tensor:
        # Return pixel-space boxes: fixed 224 scale or per-image dynamic scale.
        features = self.encoder(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        coords = self.regression_head(flattened)

        if image_size is None:
            return coords * 224

        scale = torch.stack(
            (image_size[:, 0], image_size[:, 1], image_size[:, 0], image_size[:, 1]),
            dim=1,
        )
        return coords * scale