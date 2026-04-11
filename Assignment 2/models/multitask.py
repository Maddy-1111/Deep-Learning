"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        gdown = __import__("gdown")
        gdown.download(id="11BjA_4bay8B9V0XF9IuV7eH9HSI-ufP8", output=classifier_path, quiet=False)
        gdown.download(id="1Z0LUjCHvZZYYVUgG66H7Xn9Bs8aq4V1p", output=localizer_path, quiet=False)
        gdown.download(id="1WoiNaWDCVcn2ab6VWcZ3u8k8zQnrActP", output=unet_path, quiet=False)

        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        unet.load_state_dict(torch.load(unet_path, map_location="cpu"))

        # Shared encoder and task-specific heads
        self.encoder = classifier.encoder

        self.cls_avgpool = classifier.avgpool
        self.cls_head = classifier.classifier

        self.loc_avgpool = localizer.avgpool
        self.loc_head = localizer.regression_head

        self.up4 = unet.up4
        self.up3 = unet.up3
        self.up2 = unet.up2
        self.up1 = unet.up1
        self.up_conv = unet.up_conv
        self.final_conv = unet.final_conv
        self.seg_dropout = unet.dropout

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, skips = self.encoder(x, return_features=True)

        cls_logits = self.cls_head(torch.flatten(self.cls_avgpool(bottleneck), 1))

        # Localizer head predicts normalized coordinates; convert to 224x224 pixel space.
        loc_boxes = self.loc_head(torch.flatten(self.loc_avgpool(bottleneck), 1)) * 224

        d4 = self.up4(torch.cat([self.up_conv(bottleneck), skips["skip4"]], dim=1))
        d3 = self.up3(torch.cat([d4, skips["skip3"]], dim=1))
        d2 = self.up2(torch.cat([d3, skips["skip2"]], dim=1))
        d1 = self.up1(torch.cat([d2, skips["skip1"]], dim=1))
        seg_logits = self.final_conv(self.seg_dropout(d1))

        return {
            "classification": cls_logits,
            "localization": loc_boxes,
            "segmentation": seg_logits,
        }
