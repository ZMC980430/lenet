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


def train(net: nn.Module):
    net.apply(init_weights)
    net.to(args.device)
    drawer = Drawer(legend=['trainacc', 'testacc'], figsize=(6, 6))
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
    pass
