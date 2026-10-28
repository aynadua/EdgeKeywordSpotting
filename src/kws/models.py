"""DS-CNN: the depthwise-separable CNN used for microcontroller keyword spotting.

From Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" (2017),
which remains the reference architecture for this task because it hits the
accuracy/size trade-off that MCU deployment actually needs.

The idea is a standard one: a full 3x3 convolution over C channels costs
``9*C*C`` multiplies per position, while a depthwise 3x3 followed by a 1x1
pointwise costs ``9*C + C*C``. At 64 channels that is 36,864 against 4,672 --
roughly eight times cheaper for a small accuracy cost. On a device where the
inference budget is measured in milliwatts, that trade is obviously worth
making.

The first convolution is deliberately not separable. It spans a wide slice of
the MFCC axis so the network starts from something spectrally meaningful,
before the separable stack does the cheap work.
"""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3, then pointwise 1x1, each with batch norm and ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        return self.activation(self.pointwise_norm(self.pointwise(x)))


class DSCNN(nn.Module):
    """Keyword spotter over an MFCC matrix shaped ``(1, n_frames, n_mfcc)``."""

    def __init__(
        self, n_classes: int = 12, channels: int = 64, n_blocks: int = 4, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableBlock(channels) for _ in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x))
        return self.head(self.pool(z).flatten(1))


class MFCCLogisticRegression(nn.Module):
    """The baseline: linear classifier on time-pooled MFCC statistics.

    Present so the DS-CNN has something to beat. If a linear model on the mean
    and standard deviation of each coefficient gets most of the way there, the
    convolutions are not doing much and should be justified or dropped.
    """

    def __init__(self, n_mfcc: int = 10, n_classes: int = 12) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * n_mfcc, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, frames, n_mfcc) -> pool over the time axis
        pooled = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=2).flatten(1)
        return self.linear(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
