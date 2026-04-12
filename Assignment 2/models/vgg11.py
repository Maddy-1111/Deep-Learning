import torch
import torch.nn as nn
from typing import Dict, Tuple, Union


class VGG11Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        pretrained_path: str = None,
        fine_tune: str = 'full',
        use_batchnorm: bool = True,
    ):
        # Build a VGG11-style encoder with optional pretrained loading and fine-tuning control.
        super().__init__()
        self.use_batchnorm = use_batchnorm

        block_spec = [
            (in_channels, 64, 1),
            (64, 128, 1),
            (128, 256, 2),
            (256, 512, 2),
            (512, 512, 2),
        ]

        built_blocks = [self._make_block(in_ch, out_ch, num_convs) for in_ch, out_ch, num_convs in block_spec]
        self.block1, self.block2, self.block3, self.block4, self.block5 = built_blocks
        self.blocks = nn.ModuleList(built_blocks)

        built_pools = [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(5)]
        self.pool1, self.pool2, self.pool3, self.pool4, self.pool5 = built_pools
        self.pools = nn.ModuleList(built_pools)

        if pretrained_path:
            self._load_weights(pretrained_path)

        if fine_tune == 'strict':
            for param in self.parameters():
                param.requires_grad = False
        elif fine_tune == 'partial':
            for block in self.blocks[:3]:
                for param in block.parameters():
                    param.requires_grad = False
        elif fine_tune == 'full':
            pass

    def _load_weights(self, path):
        # Load only encoder keys from a checkpoint and map them to this module.
        state_dict = torch.load(path)
        encoder_dict = {
            key.replace('encoder.', ''): value
            for key, value in state_dict.items()
            if key.startswith('encoder.')
        }
        self.load_state_dict(encoder_dict)

    def _make_block(self, in_ch, out_ch, num_convs):
        # Create one VGG stage containing repeated conv-(bn)-relu units.
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Return final encoder output, or pre-pool bottleneck plus skip features when requested.
        features = {}

        for stage_idx, (block, pool) in enumerate(zip(self.blocks, self.pools), start=1):
            x = block(x)
            if stage_idx < 5:
                features[f"skip{stage_idx}"] = x
            else:
                bottleneck = x
            x = pool(x)

        if return_features:
            return bottleneck, features
        return x