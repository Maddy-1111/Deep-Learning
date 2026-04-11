import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11UNet(nn.Module):
    def __init__(self, pretrained_path: str = None, fine_tune: str = 'full', num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, pretrained_path=pretrained_path, fine_tune=fine_tune)
        
        # Upsampling blocks
        # input channels = (upsampled_channels + skip_connection_channels)
        self.up4 = self._decoder_block(512 + 512, 512)
        self.up3 = self._decoder_block(512 + 256, 256)
        self.up2 = self._decoder_block(256 + 128, 128)
        self.up1 = self._decoder_block(128 + 64, 64)
        
        self.up_conv = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.dropout = CustomDropout(p=dropout_p)

    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encoder pass with skip connections
        bottleneck, skips = self.encoder(x, return_features=True)
        
        # 2. Decoder pass with concatenations
        # skips: skip4(512), skip3(256), skip2(128), skip1(64)
        d4 = self.up4(torch.cat([self.up_conv(bottleneck), skips["skip4"]], dim=1))
        d3 = self.up3(torch.cat([d4, skips["skip3"]], dim=1))
        d2 = self.up2(torch.cat([d3, skips["skip2"]], dim=1))
        d1 = self.up1(torch.cat([d2, skips["skip1"]], dim=1))
        
        d1 = self.dropout(d1)
        return self.final_conv(d1)