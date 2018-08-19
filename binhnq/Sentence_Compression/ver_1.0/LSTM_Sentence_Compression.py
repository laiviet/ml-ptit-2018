from __future__ import print_function
from DataLoader import *
from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from models import LSTMnet
import bcolz
import rouge
import argparse
import pickle

def load_glove():
    glove_path = "/home/binhnguyen/Downloads/glove.6B/"
    vectors = bcolz.open(glove_path + '6B.100.dat')[:]
    words = pickle.load(open(glove_path + '6B.100_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path + '6B.100_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    glove['<unk>'] = np.random.normal(scale=0.6, size=(100,))
    print("load data done!")
    return glove

def main(args=None):
    log = open('/home/binhnguyen/PycharmProjects/Sentence_Compression/ver_1.0/log.txt','w')

    BATCH = 64
    SENTENCE_LEN = 50
    glove_model = load_glove()
    train, test, valid = SC("train",SENTENCE_LEN,glove_model) , SC("test",SENTENCE_LEN,glove_model), SC("valid",SENTENCE_LEN,glove_model)
    train_data = DataLoader(train, batch_size= BATCH, shuffle=True)
    test_data = DataLoader(test , batch_size=BATCH , shuffle=False)
    valid_data = DataLoader(valid , batch_size=BATCH,shuffle=False)

    model = LSTMnet(input_dim= 100, hidden_dim= 128 )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    print("Load model done!!")

    for epoch in range(20):
        train_loss = 0.0
        for data in train_data:
            feature, label = data
            label = label.view(-1)
            feature, label = Variable(feature.float()), Variable(label.squeeze())
            y_hat = model(feature)
            loss = criterion(y_hat, label)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in test_data:
                feature, label = data
                label = label.view(-1)
                feature, label = Variable(feature.float()), Variable(label.squeeze())
                outputs = model(feature)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                test_loss += criterion(outputs , label).data.item()

        test_accuracy = correct * 100.0 / total

        correct = 0
        total = 0
        valid_loss = 0
        with torch.no_grad():
            for data in valid_data:
                feature, label = data
                label = label.view(-1)
                feature, label = Variable(feature.float()), Variable(label.squeeze())
                outputs = model(feature)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                valid_loss += criterion(outputs, label).data.item()
        valid_accuracy = correct * 100.0 / total

        print('Epoch {}'.format(epoch), file=log)
        print('Train Loss = %0.2f'%(train_loss), file = log)
        print("Test Loss = %0.2f .. Accuracy Test = %0.2f" % (test_loss, test_accuracy) , file = log)
        print('Valid Loss = %0.2f .. Accuracy Valid = %0.2f' % (valid_loss , valid_accuracy) , file = log)
        print("--------------------",file=log)

    log.close()

main()
