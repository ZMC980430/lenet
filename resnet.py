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
        Y = self.bn2(self.conv2(Y))

        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self):
        '''
        input size: 1*224*224
        '''
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )

        b3 = nn.Sequential(
            Residual(64, 128, stride=2),
            Residual(128, 128)
        )

        b4 = nn.Sequential(
            Residual(128, 256, stride=2),
            Residual(256, 256)
        )

        b5 = nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512)
        )

        b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        self.net = nn.Sequential(b1, b2, b3, b4, b5, b6)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    X = torch.ones((1, 1, 224, 224))
    resnet = ResNet()
    print(resnet(X).shape)
