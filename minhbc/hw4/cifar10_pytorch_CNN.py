import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import pickle

#class read dataset
class ReadCifar(torch.utils.data.Dataset):
    root = 'data/cifar-10-batches-py/{}'
    train_list = ['data_batch_1',
                  'data_batch_2',
                  'data_batch_3',
                  'data_batch_4']
    valid_list = ['data_batch_5']
    test_list = ['test_batch']
    def __init__(self, type = 0, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.type = type
        self.do()
    
    def do(self):
        if self.type == 0:
            self.train_data = []
            self.train_labels = []
            for f in self.train_list:
                file = self.root.format(f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    self.train_labels += entry['labels']
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape(40000, 3, 32, 32)
        elif self.type == 1:
            f = self.test_list[0]
            file = self.root.format(f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.test_data = entry['data']
                self.test_labels = entry['labels']
            self.test_data = self.test_data.reshape(10000, 3, 32, 32)
        else:
            f = self.valid_list[0]
            file = self.root.format(f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.valid_data = entry['data']
                self.valid_labels = entry['labels']
            self.valid_data = self.valid_data.reshape(10000, 3, 32, 32)
        
    def __getitem__(self, index):
        if self.type == 0:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.type == 1:
            img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.valid_data[index], self.valid_labels[index]
        return img, target
    def __len__(self):
        if self.type == 0:
            return len(self.train_data)
        elif self.type == 1:
            return len(self.test_data)
        else:
            return len(self.valid_data)

#read dataset into train/valid/test and shuffle
train_set = ReadCifar(0)
test_set = ReadCifar(1)
valid_set = ReadCifar(2)

trainloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
validloader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=4)

#get the numbers of class
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_of_classes = len(classes)


#Convolutional Neural Network (alexnet)
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,96,stride=1,padding=5,kernel_size=5,groups=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(96,256,stride=1,padding=2,kernel_size=5,groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(256,384,stride=1,kernel_size=3,padding=1,groups=1),
            nn.ReLU(),
            nn.Conv2d(384,384,stride=1,kernel_size=3,padding=1,groups=2),
            nn.ReLU(),
            nn.Conv2d(384,256,stride=1,kernel_size=3,padding=1,groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*4*4, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, num_of_classes)
        )
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 256*4*4)
        x = self.fc(x)
        return x

#init a CNN and hypeparameters  
model = Alexnet()
lr = 0.001
epoch = 5


#check torch use CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
model.to(device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


#train/test/valid
for e in range(epoch):
    # Train
    train_loss = 0.0
    for feature, label in tqdm(trainloader, desc="Training"):
        feature, label = Variable(feature.float()).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        train_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Test
    test_loss = 0.0
    y = []
    outputs = []
    for feature, label in tqdm(testloader, desc="Testing"):
        y += label.numpy().tolist()
        feature, label = Variable(feature.float()).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        test_loss += loss.data.item()
        y_hat = np.argmax(y_hat.data, axis=1)
        outputs += y_hat.tolist()
    test_acc = accuracy_score(y, outputs)
    
    # Valid
    valid_loss = 0.0
    y = []
    outputs = []
    for feature, label in tqdm(validloader, desc="Valid"):
        y += label.numpy().tolist()
        feature, label = Variable(feature.float()).to(device), Variable(label.squeeze()).to(device)
        y_hat = model(feature)
        loss = criterion(y_hat, label)
        valid_loss += loss.data.item()
        y_hat = np.argmax(y_hat.data, axis=1)
        outputs += y_hat.tolist()
    valid_acc = accuracy_score(y, outputs)
#print the result
print("Epoch: {}".format(e + 1))
    print('Train loss: {}'.format(train_loss))
    print('Test loss: {} | Test accuracy: {}'.format(test_loss, test_acc))
    print('Valid loss: {} | Valid accuracy: {}'.format(valid_loss, valid_acc))
    print('------------------------------------------------')


    