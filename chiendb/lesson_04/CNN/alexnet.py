import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from  sklearn.metrics import accuracy_score


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.num_classes = 10
        self.features = nn.Sequential(
            # input layer: 32*32*3
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16*16*64
            nn.MaxPool2d(kernel_size=2),
            # 8*8*64
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # 8*8*192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8*8*384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8*8*384
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
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


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

cifar_trainset = datasets.CIFAR10(root='/home/chiendb/Data', train=True, download=True, transform=transform)
cifar_testset = datasets.CIFAR10(root='/home/chiendb/Data', train=False, download=True, transform=transform)
train_data = DataLoader(cifar_trainset, batch_size=64, shuffle=True, num_workers=2)
test_data = DataLoader(cifar_testset, batch_size=64, shuffle=True, num_workers=2)

model = AlexNet()
lr = 0.005
num_epoch = 70
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# training
for e in range(num_epoch):
    l = 0
    for i, data in enumerate(train_data, 0):
        features, target = data
        if torch.cuda.is_available():
            features, target = features.cuda(async=True), target.cuda(async=True)
        features = Variable(features)
        target = Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss.item()
    log = open('log.txt', 'a')
    print >>log, 'epoch: {}, loss: {}'.format(e, l)
    log.close()

torch.save(model, 'alexnet.pt')
print 'Done!'

# testing
y = []
output = []
model = model.cpu()
for data in test_data:
    features, target = data
    y += target.numpy().tolist()
    y_hat = model(features)
    _, y_pred = torch.max(y_hat, 1)
    y_pred = y_pred.numpy().tolist()
    output += y_pred
print('acc: {}', accuracy_score(y, output))
