from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models.deeplabhead import DeepLabHead
from models.encoder import Encoder


class SegmentationModel(nn.Module):
    def __init__(self) -> None:
        super(SegmentationModel, self).__init__()
        self.backbone = Encoder()
        self.classifier = DeepLabHead()

    def forward(self, x1: Tensor, x2: Tensor) -> Dict[str, Tensor]:
        input_shape = x1.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x1, x2)

        x = self.classifier(features)
        x = F.interpolate(
            x, size=input_shape, mode="bilinear", align_corners=False
        )

        return x

    def train(self, mode: bool = True):
        self.backbone.train(mode)
        self.classifier.train(mode)


if __name__ == "__main__":
    x1 = torch.randn((2, 3, 256, 256))
    x2 = torch.randn((2, 1, 256, 256))
    model = SegmentationModel()
    y = model(x1, x2)
    assert y.shape == (2, 21, 256, 256)
