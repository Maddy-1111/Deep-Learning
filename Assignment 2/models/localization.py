import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    def __init__(self, pretrained_path: str = None, fine_tune: str = 'full', in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, pretrained_path=pretrained_path, fine_tune=fine_tune)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.regression_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4),
            nn.Sigmoid() # Squish to [0, 1] for relative coordinates
        )

    def forward(self, x: torch.Tensor, image_size: torch.Tensor = None) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get normalized coordinates [0, 1]
        coords = self.regression_head(x)
        
        if image_size is None:
            return coords * 224

        scale = torch.stack([image_size[:, 0], image_size[:, 1], image_size[:, 0], image_size[:, 1]], dim=1)
        return coords * scale