# SimpleCNN for UrbanSound8K — the REAL EE541 best model (E2/E3), as-shipped.
# 3 conv blocks: Conv->BN->ReLU->MaxPool x2, then Conv->BN->ReLU (no pool) -> AdaptiveAvgPool -> classifier.
# forward() returns ONLY logits (B,10). The 256-D embedding lives in get_penultimate() and is NOT served.
# Input contract for serving: (B, 1, 128, 128) log-mel spectrogram -> (B, 10) class logits.

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool (optional)"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """3-block CNN, input (B,1,128,128), output (B,10) logits"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),                    # (B,32,64,64)
            ConvBlock(32, 64),                   # (B,64,32,32)
            ConvBlock(64, 128, pool=False),      # no pool here
            nn.AdaptiveAvgPool2d((4, 4)),        # (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def get_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """256-dim activations before the final classifier, used for t-SNE (not served)."""
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        return torch.relu(self.classifier[1](flat))
