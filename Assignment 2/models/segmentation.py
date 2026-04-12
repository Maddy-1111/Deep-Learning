import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11UNet(nn.Module):
    def __init__(
        self,
        pretrained_path: str = None,
        fine_tune: str = "full",
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # Encoder
        self.backbone = VGG11Encoder(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            fine_tune=fine_tune,
        )

        # Decoder blocks
        self.dec4 = self._make_block(1024, 512)
        self.dec3 = self._make_block(768, 256)
        self.dec2 = self._make_block(384, 128)
        self.dec1 = self._make_block(192, 64, do_upsample=False)

        # Initial upsampling from bottleneck
        self.bottleneck_up = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        # Output layer
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # Dropout
        self.drop = CustomDropout(p=dropout_p)

    def _make_block(self, in_channels, out_channels, do_upsample=True):
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if do_upsample:
            block.append(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward
        bottleneck, features = self.backbone(x, return_features=True)

        # Decoder with skip connections
        x = self.bottleneck_up(bottleneck)
        x = torch.cat((x, features["skip4"]), dim=1)
        x = self.dec4(x)

        x = torch.cat((x, features["skip3"]), dim=1)
        x = self.dec3(x)

        x = torch.cat((x, features["skip2"]), dim=1)
        x = self.dec2(x)

        x = torch.cat((x, features["skip1"]), dim=1)
        x = self.dec1(x)

        x = self.drop(x)
        x = self.classifier(x)

        return x