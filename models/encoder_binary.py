import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class EncoderBinary(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, True, True]
        )
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model = IntermediateLayerGetter(
            model, return_layers={"layer4": "out"}
        )
        self.dim_reduction = nn.Conv2d(2048, 1024, kernel_size=(1, 1))

    def forward(self, x):
        output = self.model(x)
        x = output["out"]
        return self.dim_reduction(x)


if __name__ == "__main__":
    x = torch.randn((2, 1, 256, 256))
    model = EncoderBinary()
    y = model(x)
    assert y.shape == (2, 1024, 32, 32)
