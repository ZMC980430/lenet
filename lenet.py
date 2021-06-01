import torch.nn as nn
import torch
import torchvision
from argparse import Namespace


# use pth file (location: d:/Anaconda/Lib/site-packages/mypkg.pth)
from utils.drawer import Drawer


args = Namespace(
    num_epochs=50,
    lr=0.1,
    workers=4,
    device=torch.device('cuda:0')
)


def accuracy(y_hat, y):
    labels = y_hat.argmax(axis=1)
    return float((labels.type(y.dtype) == y).sum())  # faster than mask


def evaluate(net, testiter):
    net.eval()
    acc, total = 0., 0.
    for x, y in testiter:
        x, y = x.to(args.device), y.to(args.device)
        y_hat = net(x)
        acc += accuracy(y_hat, y)
        total += len(y)
    return float(acc/total)


# print outputs shape of every layer
def checkshape(net, dataiter):
    data, _ = next(dataiter)
    print(data.shape)
    for layer in net:
        print(layer)
        data = layer(data)
        print(data.shape)
    print(data)


def train_epoch(net, trainset, optimizer, criterion,
                drawer=None, testset=None, epoch=0):

    net.train()
    trainiter = iter(torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size=256))
    testiter = iter(
        torch.utils.data.DataLoader(testset, shuffle=False, batch_size=256))
    trainacc, testacc, total = 0., 0., 0.

    for x, y in trainiter:
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        y_hat = net(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total += len(y)
        trainacc += accuracy(y_hat, y)
    trainacc = trainacc / total
    if testiter is not None:
        testacc = evaluate(net, testiter)
    if drawer is not None:
        drawer.add(epoch, (trainacc, testacc))
    print(f'epoch: {epoch:02d},\
        loss: {loss:.4f},\
        trainacc: {trainacc:.4f},\
        testacc: {testacc:.4f}')


def train(net):

    # get FashionMNIST
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', transform=torchvision.transforms.ToTensor(), train=True
    )
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=torchvision.transforms.ToTensor()
    )

    drawer = Drawer(
        legend=['trainacc', 'testacc'],
        ylim=[0, 1],
        xlim=[0, args.num_epochs],
        figname='LeNet'
    )
    net.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        train_epoch(net, trainset, optimizer, criterion, drawer=drawer,
                    testset=testset, epoch=epoch)
    drawer.show()


if __name__ == '__main__':
    # LeNet
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    # input shape: 1*28*28
    LeNet1 = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=3, padding=1),  # 6*28*28
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1),      # 6*26*26
        nn.Conv2d(6, 16, kernel_size=3),            # 16*24*24
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),      # 16*12*12
        nn.Conv2d(16, 20, kernel_size=3),           # 20*10*10
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),      # 20*5*5
        nn.Flatten(),                               # 500
        nn.Linear(500, 120),
        nn.ReLU(),
        nn.Linear(120, 60),
        nn.ReLU(),
        nn.Linear(60, 10)
    )
    train(LeNet1)
    # testset = torchvision.datasets.FashionMNIST(
    #     root='./data', train=False,
    #     transform=torchvision.transforms.ToTensor()
    # )
    # testiter = iter(
    #     torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1)
    # )
    # checkshape(LeNet1, testiter)
