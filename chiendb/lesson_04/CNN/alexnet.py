import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import CIFAR10
from torch.utils.data import DataLoader
from  sklearn.metrics import accuracy_score


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.num_classes = 10
        self.features = nn.Sequential(
            # input layer: 32*32*3
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16*16*96
            nn.MaxPool2d(kernel_size=2),
            # 8*8*96
            nn.Conv2d(96, 192, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # 8*8*192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8*8*384
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # 8*8*384
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # 8*8*256
            nn.MaxPool2d(kernel_size=2)
            # 4*4*256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*4*256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4*4*256)
        x = self.classifier(x)
        return x


train_data = CIFAR10(0)
valid_data = CIFAR10(1)
test_data = CIFAR10(2)
train_data = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
valid_data = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=2)
test_data = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

model = AlexNet()
lr = 0.005
num_epoch = 60
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=lr)

# -------------------training----------------------
for e in range(num_epoch):
    log = open('log.txt', 'a')
    print('------------------- Epoch: {}-----------------------'.format(e), file=log)
    l = 0
    for data in train_data:
        features, target = data
        if torch.cuda.is_available():
            features, target = features.float().cuda(async=True), target.long().cuda(async=True)
        features = Variable(features)
        target = Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss.item()

    print('train: loss = {}'.format(l), file=log)

    # -----------------valid------------------------
    l = 0
    y = []
    output = []
    for data in valid_data:
        features, target = data
        y += target.numpy().tolist()
        if torch.cuda.is_available():
            features, target = features.float().cuda(async=True), target.long().cuda(async=True)
        features = Variable(features)
        target = Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.cpu().numpy().tolist()
        output += y_pred
        l += loss.item()

    acc = accuracy_score(y, output)
    print('valid: loss = {}, acc = {}'.format(l, acc), file=log)

    # -----------------test-------------------------
    l = 0
    y = []
    output = []
    for data in valid_data:
        features, target = data
        y += target.numpy().tolist()
        if torch.cuda.is_available():
            features, target = features.float().cuda(async=True), target.long().cuda(async=True)
        features = Variable(features)
        target = Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.cpu().numpy().tolist()
        output += y_pred
        l += loss.item()

    acc = accuracy_score(y, output)
    print('valid: loss = {}, acc = {}'.format(l, acc), file=log)

    log.close()

torch.save(model, 'alexnet.pt')

print('Done')