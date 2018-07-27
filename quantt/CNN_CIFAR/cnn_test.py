from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import pickle
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR

class LoadData():
    def __init__(self):
        self.trainZip = list()
        path = '/home/student/hientm_folder/jupyter/data/cifar-10-batches-py'
        temp = self.unpickle(os.path.join(path, 'data_batch_' + str(1)))
        self.trainZip.append(np.array(temp[b'data']))
        self.trainZip.append(np.array(temp[b'labels']))
        for i in range(2, 5):
            temp = self.unpickle(os.path.join(path, 'data_batch_' + str(i)))
            x = np.array(temp[b'data'])
            y = np.array(temp[b'labels'])
            self.trainZip[0] = np.vstack((self.trainZip[0], x))
            self.trainZip[1] = np.hstack((self.trainZip[1], y))
        temp = self.unpickle(os.path.join(path, 'data_batch_' + str(5)))
        x = np.array(temp[b'data'])
        y = np.array(temp[b'labels'])
        self.valid = [x, y]
        temp = self.unpickle(os.path.join(path, 'test_batch'))
        x = np.array(temp[b'data'])
        y = np.array(temp[b'labels'])
        self.test = [x, y]
        self.index = 0
    def getall(self):
        return self.trainZip,self.valid,self.test

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
class CusData(Dataset):
    def __init__(self,data):
        # stuff
        super(CusData,self).__init__()
        self.data=data
    def __getitem__(self, index):
        # stuff
        return self.data[0][index],self.data[1][index]
    def __len__(self):
        return len(self.data[0])
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.FullyConnectedLayers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), -1)
        x = self.FullyConnectedLayers(x)
        return x
model = AlexNet()
lr = 0.0001
epochs = 20
transform = transforms.Compose(
    [transforms.ToTensor()]
     )
train,valid,test=LoadData().getall()
train[0]=train[0].reshape((-1,3,32,32))
valid[0] = valid[0].reshape((-1, 3, 32, 32))
test[0] = test[0].reshape((-1, 3, 32, 32))
train[0]=torch.from_numpy(train[0]).type(torch.FloatTensor)
train[1] = torch.from_numpy(train[1]).type(torch.LongTensor)
valid[0] = torch.from_numpy(valid[0]).type(torch.FloatTensor)
valid[1] = torch.from_numpy(valid[1]).type(torch.LongTensor)
test[0]=torch.from_numpy(test[0]).type(torch.FloatTensor)
test[1] = torch.from_numpy(test[1]).type(torch.LongTensor)
train=CusData(train)
valid = CusData(valid)
test = CusData(test)
trainloader = DataLoader(train, batch_size=64,shuffle=True)
validloader = DataLoader(valid, batch_size=64,shuffle=False)
testloader = DataLoader(test, batch_size=64,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cpu()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
logs=[]
scheduler = MultiStepLR(optimizer, milestones=[15], gamma=0.1)
for epoch in range(epochs):
    scheduler.step()
    train_loss = 0.0
    model.train()
    for feature, label in trainloader:
        feature, label = Variable(feature).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_loss = 0.0
    output = []
    y = []
    model.eval()
    for feature, label in validloader:
        y += label.numpy().tolist()
        feature, label = Variable(feature).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        val_loss += loss.item()
        y_hat = np.argmax(y_hat.data, axis=1)
        output += y_hat.tolist()
    val_acc = accuracy_score(y, output)
    test_loss = 0.0
    y = []
    output = []
    for feature, label in testloader:
        y += label.numpy().tolist()
        feature, label = Variable(feature).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        test_loss += loss.item()
        y_hat = np.argmax(y_hat.data, axis=1)
        output += y_hat.tolist()
    test_acc = accuracy_score(y, output)
    log='Epoch %2d: loss: %10.2f Valid: loss: %5.2f acc: %0.4f Test: loss: %5.2f acc: %0.4f' %(epoch, train_loss, val_loss, val_acc, test_loss, test_acc)
    print(log)
    logs.append(log)
thefile = open('/home/student/quantt/train_result.txt', 'w')
for item in log:
    thefile.write("%s\n" % item)

