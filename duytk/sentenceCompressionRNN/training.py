from __future__ import print_function
from dataloader import *

from torch.utils.data import DataLoader
import argparse
import torch
from models import RNNModule
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def getExY(Y , b ,slen):
    ans = []
    for ib in range(b):
        le = slen[ib]
        for il in range(50):
            if il <le :
                ans.append(Y[ib * 50 + il])
    return ans

def main(args):
    log = open("log1.txt", "a")
    log.truncate(0)

    train , test , valid = SC_DATA("train") , SC_DATA("test") , SC_DATA("valid")
    print('LOAD WORD2VEC VOCAB DONE !! =========================')
    train_dl = DataLoader(train , batch_size=args.batch , shuffle=True)
    test_dl  = DataLoader(test  , batch_size=args.batch , shuffle=False)
    valid_dl = DataLoader(valid , batch_size=args.batch , shuffle=False)

    # model
    model = RNNModule( input_dim= 100, hidden_dim= 128 ,lstm=True).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(args.epoch):
        # Train
        train_loss = 0.0
        for feature, label , slen  in tqdm(train_dl, desc='Training', leave=False):
            # print(label)
            #print(type(feature))
            label = label.view(-1)
            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature , slen)
            loss = criterion(y_hat, label)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        test_loss = 0.0
        y = []
        output = []
        for feature, label , slen in tqdm(test_dl, desc='Testing', leave=False):
            label = label.view(-1)
            y += getExY( label.numpy().tolist() , len(slen), slen)

            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature , slen)
            loss = criterion(y_hat, label)
            test_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data, axis=1) # b x len
            output +=  getExY( y_hat.tolist() , len(slen) , slen)

        test_acc = accuracy_score(y, output)

        # Valid
        valid_loss = 0.0
        y = []
        output = []
        for feature, label , slen in tqdm(valid_dl, desc='Valid', leave=False):
            label = label.view(-1)
            y += getExY( label.numpy().tolist() , len(slen) , slen)
            feature, label = Variable(feature.float()).cuda(), Variable(label.squeeze()).cuda()
            y_hat = model(feature , slen)
            loss = criterion(y_hat, label)
            valid_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data, axis=1)
            output +=  getExY( y_hat.tolist() , len(slen) , slen)
        valid_acc = accuracy_score(y, output)

        print('Epoch %2d: loss: %10.2f > Test loss: %5.2f acc: %0.4f > Valid loss: %5.2f acc: %0.4f ' % (epoch, train_loss, test_loss, test_acc, valid_loss, valid_acc), file=log)
        print()
        print('Epoch %2d: loss: %10.2f > Test loss: %5.2f acc: %0.4f > Valid loss: %5.2f acc: %0.4f ' % (epoch, train_loss, test_loss, test_acc, valid_loss, valid_acc))
    torch.save(model, 'mymodel.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()

main(args)
