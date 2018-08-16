from dataloader import Loader
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from models import SentenceCompression
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------------------ #
# read data
train_data = Loader(0)
valid_data = Loader(1)
test_data = Loader(2)
train_data = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
valid_data = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4)
test_data = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

# ------------------------------------------------------------------------------ #
# read word2vec
with open('/home/student/glove/glove.6B.100d.txt', 'r') as f:
    lines = f.readlines()

dictionary = {}
weight_matrix = []
for i, line in enumerate(lines):
    word = line.split()
    dictionary[word[0]] = i
    weight_matrix.append([float(word[i]) for i in range(1, len(word))])

weight_matrix = np.array(weight_matrix)

# ------------------------------------------------------------------------------- #
model = SentenceCompression(100, 100, 50, 2, weight_matrix, dictionary)
lr = 0.01
num_epoch = 20

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

print('Done load_model')

for e in range(num_epoch):
    log = open('log.txt', 'a')
    print('------------------- Epoch: {}-----------------------'.format(e), file=log)
    l = 0
    for data in train_data:
        features, target = data
        print(len(features))
        print(len(target))
        target = torch.Tensor(target).long()
        target = Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss.item()
        print('Done loss {}'.format(loss.item))

    print('train: loss = {}'.format(l), file=log)

    # -----------------valid------------------------
    l = 0
    y = []
    output = []
    for data in valid_data:
        features, target = data
        features, target = torch.Tensor(features), torch.Tensor(target).long()
        y += target.numpy().tolist()
        features, target = Variable(features), Variable(target)
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
        features, target = torch.Tensor(features), torch.Tensor(target).long()
        y += target.numpy().tolist()
        features, target = Variable(features), Variable(target)
        y_hat = model(features)
        loss = criterion(y_hat, target)
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.cpu().numpy().tolist()
        output += y_pred
        l += loss.item()

    acc = accuracy_score(y, output)
    print('valid: loss = {}, acc = {}'.format(l, acc), file=log)

    log.close()

torch.save(model, '/data01/orm/flask_app/model/classifier.pt')

print('Done')