import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import LoadData


class Classifier(nn.Module):
    def __init__(self, h, n_features, n_classes):
        super(Classifier, self).__init__()
        self.rl = nn.ReLU()
        self.l1 = nn.Linear(n_features, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, n_classes)

    def forward(self, X):
        X = self.l1(X)
        X = self.rl(X)
        X = self.l2(X)
        X = self.rl(X)
        X = self.l3(X)
        return X


epoch = 20000
h = 100
n_features = 3072
n_classes = 10
lr = 0.0001
load = LoadData()
model = Classifier(h=h, n_features=n_features, n_classes=n_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

log = []
for e in range(epoch):
    X_batch, y_batch = load.get_batch()
    features = Variable(torch.from_numpy(X_batch).type(torch.FloatTensor))
    target = Variable(torch.from_numpy(y_batch).type(torch.LongTensor))
    y_hat = model(features)
    loss = criterion(y_hat, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 1000 == 0:
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.data.numpy()
        acc = accuracy_score(y_batch, y_pred)
        print(acc)
        # p, r, f, _ = precision_recall_fscore_support(y_batch, y_pred)
        # cm = confusion_matrix(y, y_pred)
        # log.append((e, loss.data[0], f[0], f[1]))
    if e % 1000 == 0:
        print('Epoch %d: %f' % (e, loss))
print('DONE')

"""
epochs, losses, f0, f1 = zip(*log)
figure = plt.plot(epochs, losses, 'r-', epochs, f0, 'b-', epochs, f1, 'g-')
plt.show()
"""