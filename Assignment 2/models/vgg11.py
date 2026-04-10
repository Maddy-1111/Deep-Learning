import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # VGG11 Configuration: (channels, num_convs)
        self.block1 = self._make_block(in_channels, 64, 1)
        self.block2 = self._make_block(64, 128, 1)
        self.block3 = self._make_block(128, 256, 2)
        self.block4 = self._make_block(256, 512, 2)
        self.block5 = self._make_block(512, 512, 2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_block(self, in_ch, out_ch, num_convs):
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        features = {}
        
        # Process blocks and save features before pooling for U-Net skip connections
        x = self.block1(x)
        features["skip1"] = x
        x = self.pool(x)
        
        x = self.block2(x)
        features["skip2"] = x
        x = self.pool(x)
        
        x = self.block3(x)
        features["skip3"] = x
        x = self.pool(x)
        
        x = self.block4(x)
        features["skip4"] = x
        x = self.pool(x)
        
        x = self.block5(x)
        # x here is the bottleneck feature map
        
        if return_features:
            return x, features
        return x