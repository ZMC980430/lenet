import torch
import torch.nn.functional as F
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, padding=1)

        if in_channel != out_channel:
            self.conv3 = nn.Conv2d(in_channel, out_channel,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))

        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return Y


if __name__ == "__main__":
    X = torch.ones((1, 3, 96, 96))
    res = Residual(3, 16, stride=2)
    print(res(X).shape)
