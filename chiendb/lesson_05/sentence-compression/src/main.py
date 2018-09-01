from dataloader import Loader
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from models import SentenceCompression
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def delete_excess(input, lens, max_len):
    output = []
    for i in range(len(input)//max_len):
        for j in range(max_len):
            if j < lens[i]:
                output.append(input[i*max_len+j])
    return output


def adjust_learning_rate(optimizer, t):
    lr = 0.002 * (0.5 ** (t // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

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
# read data
train_data = Loader(0, dictionary)
valid_data = Loader(1, dictionary)
test_data = Loader(2, dictionary)
train_data = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
valid_data = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4)
test_data = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

# ------------------------------------------------------------------------------ #
model = SentenceCompression(100, 100, 50, 2, weight_matrix, len(dictionary)).cuda()
lr = 0.001
num_epoch = 30

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0001)

for e in range(num_epoch):
    log = open('../log/log.txt', 'a')
    print('------------------- Epoch: {}-----------------------'.format(e), file=log)
    optimizer = adjust_learning_rate(optimizer, e)
    l = 0
    y = []
    output = []
    for data in train_data:
        features, target, lengths = data
        target = target.view(-1)
        y += delete_excess(target.numpy().tolist(), lengths.numpy().tolist(), 50)
        features, target, lengths = Variable(features).cuda(), Variable(target).cuda(), Variable(lengths).cuda()
        y_hat = model(features, lengths)
        loss = criterion(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss.item()
        _, y_pred = torch.max(y_hat, 1)
        output += delete_excess(y_pred.cpu().numpy().tolist(), lengths.cpu().numpy().tolist(), 50)

    acc = accuracy_score(y, output)
    print('train: loss = {}, acc = {}'.format(l, acc), file=log)

    # -----------------valid------------------------
    l = 0
    y = []
    output = []
    for data in valid_data:
        features, target, lengths = data
        target = target.view(-1)
        y += delete_excess(target.numpy().tolist(), lengths.numpy().tolist(), 50)
        features, target, lengths = Variable(features).cuda(), Variable(target).cuda(), Variable(lengths).cuda()
        y_hat = model(features, lengths)
        loss = criterion(y_hat, target)
        _, y_pred = torch.max(y_hat, 1)
        output += delete_excess(y_pred.cpu().numpy().tolist(), lengths.cpu().numpy().tolist(), 50)
        l += loss.item()

    acc = accuracy_score(y, output)
    print('valid: loss = {}, acc = {}'.format(l, acc), file=log)

    # -----------------test-------------------------
    l = 0
    y = []
    output = []
    for data in test_data:
        features, target, lengths = data
        target = target.view(-1)
        y += delete_excess(target.numpy().tolist(), lengths.numpy().tolist(), 50)
        features, target, lengths = Variable(features).cuda(), Variable(target).cuda(), Variable(lengths).cuda()
        y_hat = model(features, lengths)
        loss = criterion(y_hat, target)
        _, y_pred = torch.max(y_hat, 1)
        output += delete_excess(y_pred.cpu().numpy().tolist(), lengths.cpu().numpy().tolist(), 50)
        l += loss.item()

    acc = accuracy_score(y, output)
    print('valid: loss = {}, acc = {}'.format(l, acc), file=log)

    log.close()

torch.save(model, '../log/model.pt')

print('Done')