import torch
import torch.nn as nn

from models.encoder_binary import EncoderBinary
from models.encoder_rgb import EncoderRGB


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_encoder = EncoderBinary()
        self.rgb_encoder = EncoderRGB()

    def forward(self, x1, x2):
        y1 = self.rgb_encoder(x1)
        y2 = self.binary_encoder(x2)
        return torch.concat([y1, y2], 1)


if __name__ == "__main__":
    x1 = torch.randn((2, 3, 256, 256))
    x2 = torch.randn((2, 1, 256, 256))
    model = Encoder()
    y = model(x1, x2)
    assert y.shape == (2, 2048, 32, 32)
    print(y.shape)
