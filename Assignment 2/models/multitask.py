import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Multi-task model with a shared encoder and separate heads."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        def _extract_state_dict(checkpoint_obj):
            if isinstance(checkpoint_obj, dict):
                for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
                    nested = checkpoint_obj.get(key)
                    if isinstance(nested, dict):
                        checkpoint_obj = nested
                        break
            return checkpoint_obj

        def _load_compatible_weights(model: nn.Module, checkpoint_path: str):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
            if not isinstance(state_dict, dict):
                return

            model_state = model.state_dict()
            compatible = {}
            for key, value in state_dict.items():
                if not torch.is_tensor(value):
                    continue
                normalized_key = key[7:] if key.startswith("module.") else key
                if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
                    compatible[normalized_key] = value

            if compatible:
                model.load_state_dict(compatible, strict=False)

        # Download weights dynamically
        downloader = __import__("gdown")
        downloader.download(id="11BjA_4bay8B9V0XF9IuV7eH9HSI-ufP8", output=classifier_path, quiet=False)
        downloader.download(id="1h4crWog-_c62D9ACIvg-YhHzcax2vYiG", output=localizer_path, quiet=False)
        downloader.download(id="1Vn_n0Tkdipho3rQ1XaIBnyHaz_QvxVgA", output=unet_path, quiet=False)

        # Instantiate task-specific models
        cls_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc_model = VGG11Localizer(in_channels=in_channels)
        seg_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load pretrained weights
        _load_compatible_weights(cls_model, classifier_path)
        _load_compatible_weights(loc_model, localizer_path)
        _load_compatible_weights(seg_model, unet_path)

        # Shared feature extractor
        self.shared_encoder = cls_model.encoder

        # Classification components
        self.cls_pool = cls_model.avgpool
        self.cls_fc = cls_model.classifier

        # Localization components
        self.loc_pool = loc_model.avgpool
        self.loc_fc = loc_model.regression_head

        # Segmentation components (decoder)
        self.seg_bottleneck_up = seg_model.bottleneck_up
        self.seg_dec4 = seg_model.dec4
        self.seg_dec3 = seg_model.dec3
        self.seg_dec2 = seg_model.dec2
        self.seg_dec1 = seg_model.dec1
        self.seg_head = seg_model.classifier
        self.seg_drop = seg_model.drop

    def forward(self, x: torch.Tensor):
        # Extract shared features
        bottleneck, skip_feats = self.shared_encoder(x, return_features=True)

        # ---- Classification branch ----
        cls_features = self.cls_pool(bottleneck)
        cls_features = torch.flatten(cls_features, start_dim=1)
        cls_out = self.cls_fc(cls_features)

        # ---- Localization branch ----
        loc_features = self.loc_pool(bottleneck)
        loc_features = torch.flatten(loc_features, start_dim=1)
        loc_out = self.loc_fc(loc_features) * 224  # scale to image space

        # ---- Segmentation branch ----
        x_seg = self.seg_bottleneck_up(bottleneck)
        x_seg = torch.cat((x_seg, skip_feats["skip4"]), dim=1)
        x_seg = self.seg_dec4(x_seg)

        x_seg = torch.cat((x_seg, skip_feats["skip3"]), dim=1)
        x_seg = self.seg_dec3(x_seg)

        x_seg = torch.cat((x_seg, skip_feats["skip2"]), dim=1)
        x_seg = self.seg_dec2(x_seg)

        x_seg = torch.cat((x_seg, skip_feats["skip1"]), dim=1)
        x_seg = self.seg_dec1(x_seg)

        x_seg = self.seg_drop(x_seg)
        seg_out = self.seg_head(x_seg)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }