import torch.nn as nn

EPS = 1e-7


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError


class Encoder32(nn.Module):
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(
                cin, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 32x32 -> 16x16
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 16x16 -> 8x8
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, nf, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 8x8 -> 4x4
            nn.GroupNorm(nf // 4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                nf, cout, kernel_size=4, stride=1, padding=0, bias=False
            ),  # 4x4 -> 1x1
        ]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)
