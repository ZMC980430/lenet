import torch
import torch.nn as nn
import torchvision
from utils.trainer import train_epoch
from utils.drawer import Drawer
from argparse import Namespace


args = Namespace(
    lr=0.05,
    num_epochs=30,
    device=torch.device('cuda:0'),
    num_workers=4
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


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


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

    # ninblock
    class ninblock(nn.Module):
        def __init__(self, input_channel, output_channel,
                     kernel_size, stride, padding):
            super().__init__()
            self.conv1 = nn.Conv2d(
                input_channel, output_channel, kernel_size=kernel_size,
                stride=stride, padding=padding)
            self.conv2 = nn.Conv2d(
                output_channel, output_channel, kernel_size=1)
            self.conv3 = nn.Conv2d(
                output_channel, output_channel, kernel_size=1)

        def forward(self, X):
            X = self.conv1(X)
            X = nn.ReLU()(X)
            X = self.conv2(X)
            X = nn.ReLU()(X)
            X = self.conv3(X)
            X = nn.ReLU()(X)
            return X

    def getninblock(in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU()
        )

    # nin
    nin = nn.Sequential(
        getninblock(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        getninblock(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        getninblock(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        getninblock(384, 10, kernel_size=3, stride=1, padding=1),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten()
    )
    nin.apply(init_weights)
    train(nin)
    # epoch, loss, trainacc, testacc = 1, 1, 1, 1
    # print(
    #     f"epoch: {epoch:02d}, loss: {loss:.4f}, "
    #     f"trainacc: {trainacc:.4f}, testacc: {testacc:.4f}"
    # )
