import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    def __init__(self, pretrained_path: str = None, freeze_encoder: bool = False, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, pretrained_path=pretrained_path, freeze=freeze_encoder)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get normalized coordinates [0, 1]
        coords = self.regression_head(x)
        
        # Scale to image pixel space (224 x 224)
        return coords * 224.0