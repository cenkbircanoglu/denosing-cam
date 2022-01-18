
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import ASPP


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int=2048, num_classes: int=21) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )