import torch
import torch.nn as nn
import torchvision
from utils.trainer import train_epoch
from utils.drawer import Drawer
from argparse import Namespace


args = Namespace(
    lr=0.01,
    num_epochs=30,
    device=torch.device('cuda:0')
    # device=None
)


def train(net: nn.Module):
    net.to(args.device)
    drawer = Drawer(legend=['trainacc', 'testacc'], figsize=(4, 4))
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
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
        figname='AlexNet'
    )

    for epoch in range(args.num_epochs):
        train_epoch(net, trainset, optimizer, criterion, drawer,
                    testset, epoch=epoch, device=args.device)
    drawer.show()


if __name__ == '__main__':
    # nvidia-smi dmon -d 3 -s pum
    # input shape: 1*224*224
    alexnet = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096),
        nn.Linear(4096, 1000),
        nn.Linear(1000, 10)
    )
    train(alexnet)
