from __future__ import print_function

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import CIFAR10
from models import *
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def main(args):
    train = CIFAR10(0)
    test  = CIFAR10(1)
    valid = CIFAR10(2)

    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(test, batch_size=args.batch * 8, shuffle=False)
    valid_dl = DataLoader(valid, batch_size=args.batch * 8, shuffle=False)

    # model = SimpleFeedForward([3072,4096,4096,10], activation=nn.LeakyReLU()).cuda()
    model = AlexNetCifa10().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        # Train
        train_loss = 0.0
        for feature, label in tqdm(train_dl, desc='Training', leave=False):
            #print(label)
            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        test_loss = 0.0
        y = []
        output = []
        for feature, label in tqdm(test_dl, desc='Testing', leave=False):
            y += label.numpy().tolist()
            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            test_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data, axis=1)
            output += y_hat.tolist()
        test_acc = accuracy_score(y, output)

        #Valid
        valid_loss = 0.0
        y = []
        output = []
        for feature, label in tqdm(valid_dl, desc='Valid', leave=False):
            y += label.numpy().tolist()
            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            valid_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data, axis=1)
            output += y_hat.tolist()
        valid_acc = accuracy_score(y, output)
        log = open("log4.txt", "a")
        print('Epoch %2d: loss: %10.2f > Test loss: %5.2f acc: %0.4f > Valid loss: %5.2f acc: %0.4f ' %(epoch, train_loss, test_loss, test_acc , valid_loss , valid_acc) , file = log)
        print('Epoch %2d: loss: %10.2f > Test loss: %5.2f acc: %0.4f > Valid loss: %5.2f acc: %0.4f ' % (epoch, train_loss, test_loss, test_acc, valid_loss , valid_acc))
    torch.save(model, 'mymodel.pt')

        #print(confusion_matrix(y, output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
main(args)