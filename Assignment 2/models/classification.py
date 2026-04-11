import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_batchnorm: bool = True, fine_tune: str = 'full'):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, fine_tune=fine_tune, use_batchnorm=use_batchnorm)
        head_norm = nn.BatchNorm1d if use_batchnorm else (lambda _: nn.Identity())
        
        # Standard VGG head: Pool to 7x7, flatten, then dense layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            head_norm(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            head_norm(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from backbone
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification logits
        logits = self.classifier(x)
        return logits