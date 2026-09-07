"""
model.py
--------
Two model families:

1. DocumentClassifier - transfer learning on ResNet18 / EfficientNet-B0,
   used for document-type classification (and can double as a genuine/
   suspicious visual classifier if you fine-tune it on a tampering dataset).

2. ConvAutoencoder - a small unsupervised anomaly detector. Trained ONLY on
   genuine documents, it learns to reconstruct them well; a document that
   reconstructs poorly (high pixel error) is flagged as visually anomalous.
   This is the "tampering/anomaly detection" stage of the pipeline and is
   intentionally simple/trainable on a laptop.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str, pretrained: bool):
    """Return (feature_extractor, feature_dim) for a given backbone name."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet34(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    raise ValueError(f"Unknown backbone: {name}")


class DocumentClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, freeze_backbone: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone, pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class ConvAutoencoder(nn.Module):
    """Lightweight autoencoder for unsupervised visual anomaly detection."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_device():
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
