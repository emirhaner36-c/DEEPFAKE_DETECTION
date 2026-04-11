"""
EfficientNet-B4 deepfake classifier via the timm library.
"""
import torch
import torch.nn as nn
import timm


class DeepfakeClassifier(nn.Module):
    """
    EfficientNet-B4 backbone with a custom binary classification head.

    The backbone is loaded with ImageNet-pretrained weights and fine-tuned
    end-to-end (transfer learning).
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnet_b4_ns",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,       # remove the default classification head
            global_pool="avg",
        )

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_feature_layer(self) -> nn.Module:
        """Returns the last convolutional block — used by Grad-CAM."""
        # Both EfficientNet-B0 and B4 expose conv_head
        return self.backbone.conv_head


def build_model(cfg: dict, device: torch.device) -> DeepfakeClassifier:
    model_cfg = cfg["model"]
    model = DeepfakeClassifier(
        backbone=model_cfg.get("backbone", "tf_efficientnet_b4_ns"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )
    return model.to(device)
