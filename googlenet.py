'''
2021/06/03
GoogLeNet
'''

from argparse import Namespace
import torch
import torch.nn as nn
import torchvision
from utils.drawer import Drawer
from utils.trainer import train_epoch


args = Namespace(
    lr=0.1,
    num_epochs=30,
    device=torch.device('cuda:0'),
    num_workers=4
    # device=None
)


def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


class Inception(nn.Module):
    def __init__(self, in_channel, o1, o2, o3, o4):
        '''
        path1: 1*1 conv, o1 requires 1 arg
        path2: 1*1 conv, 3*3 conv, o2 requires 2 args
        path3: 1*1 conv, 5*5 conv, o3 requires 2 args
        path4: MaxPool, 1*1 conv, o4 requires 1 arg
        '''
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channel, o1, kernel_size=1),
            nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channel, o2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(o2[0], o2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channel, o3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(o3[0], o3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, o4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, X):
        X1 = self.path1(X)
        X2 = self.path2(X)
        X3 = self.path3(X)
        X4 = self.path4(X)
        return torch.cat((X1, X2, X3, X4), dim=1)


def train(net: nn.Module):
    net.apply(init_weights)
    net.to(args.device)
    drawer = Drawer(legend=['trainacc', 'testacc'], figsize=(4, 4))
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((96, 96)),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', transform=transform, train=True
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=transform
    )

    drawer = Drawer(
        legend=['trainacc', 'testacc'],
        ylim=[0, 1],
        xlim=[0, args.num_epochs],
        figname='GoogLeNet'
    )

    for epoch in range(args.num_epochs):
        train_epoch(net, trainset, optimizer, criterion, drawer,
                    testset, epoch=epoch, device=args.device)
    drawer.show()


if __name__ == '__main__':
    googlenet = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
        nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
        nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
        nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),
        nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ),
        nn.Linear(1024, 10)
    )
    train(googlenet)
