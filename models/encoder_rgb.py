import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class EncoderRGB(nn.Module):
    def __init__(self):
        super().__init__()
        model1 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, True, True]
        )
        self.model1 = IntermediateLayerGetter(model1, return_layers={"layer4": "out"})
        self.dim_reduction = nn.Conv2d(2048, 1024, kernel_size=(1, 1))

    def forward(self, x):
        output = self.model1(x)
        x = output["out"]
        return self.dim_reduction(x)


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    model = EncoderRGB()
    y = model(x)
    assert y.shape == (2, 1024, 32, 32)
    print(y.shape)
