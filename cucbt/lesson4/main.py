from __future__ import print_function

import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from load_data import CIFAR10
from model import *
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def main(args):
    train = CIFAR10('train')
    test = CIFAR10('test')
    valid = CIFAR10('valid')

    train_data = DataLoader(train, batch_size=args.batch, shuffle=True)
    valid_data = DataLoader(valid, batch_size=args.batch * 8, shuffle=False)
    test_data = DataLoader(test, batch_size=args.batch * 8, shuffle=False)

    model = AlexNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        # Train
        train_loss = 0.0
        for feature, label in tqdm(train_data, desc='Training', leave=False):
            # print(label)
            feature, label = Variable(feature).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Valid
        val_loss = 0.0
        output = []
        y = []
        for feature, label in tqdm(valid_data, desc='Validation', leave=False):
            y += label.numpy().tolist()
            feature, label = Variable(feature).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            val_loss += loss.data[0]
            y_hat = np.argmax(y_hat.data, axis=1)
            output += y_hat.tolist()

        val_acc = accuracy_score(y, output)

        test_loss = 0.0
        y = []
        output = []
        for feature, label in tqdm(test_data, desc='Testing', leave=False):
            y += label.numpy().tolist()
            feature, label = Variable(feature).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            test_loss += loss.data[0]
            y_hat = np.argmax(y_hat.data, axis=1)
            output += y_hat.tolist()
        test_acc = accuracy_score(y, output)

        print('Epoch %2d: loss: %10.2f > loss: %5.2f acc: %0.4f > loss: %5.2f acc: %0.4f' % (
        epoch, train_loss, val_loss, val_acc, test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()
    main(args)
