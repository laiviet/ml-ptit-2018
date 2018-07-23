import numpy as np
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_data_train():
    data = []
    label = []
    for i in range (5) :
        file = "../cifar-10-batches-py/data_batch_"+str(i+1)
        temp =unpickle(file)
        if(i == 0):
            data = temp['data']
            label = temp['labels']
        else:
            data = np.append(data, temp['data'])
            label = np.append(label, temp['labels'])

    return data, label

def load_data_test():
    file = "../cifar-10-batches-py/test_batch"
    temp = unpickle(file)
    data = temp['data']
    label = temp['labels']
    return data, label

def scaler():
    data_train, label_train = load_data_train()
    data_test, label_test = load_data_test()

    data_train = np.reshape(data_train, (50000, 3072))

    return data_train, label_train, data_test, label_test

epoch = 100
h = 10
lr = 0.01

data_train, label_train, data_test, label_test = scaler()
# print(data_train)

# model = nn.Sequential(
#     nn.Linear(3072,h),
#     nn.Sigmoid(),
#     nn.Linear(h,h),
#     nn.Sigmoid(),
#     nn.Linear(h,10)
# )

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.l1 = nn.Linear(3072,h)
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(h,10)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.sigmoid(x)
        return x
model = Module()

optimizer = torch.optim.SGD (model.parameters(), lr=lr)
weight = torch.FloatTensor(np.array([1,2,3, 4, 5, 6, 7, 8, 9, 10]))
criterion = nn.CrossEntropyLoss(weight=weight)

features = Variable(torch.from_numpy(data_train).type(torch.FloatTensor))
target = Variable(torch.from_numpy(label_train))
log = []

for e in range(epoch):
    y_hat = model(features)
    loss = criterion(y_hat, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, y_pred = torch.max(y_hat, 1)
    y_pred = y_pred.data.numpy()

    acc = accuracy_score(label_train, y_pred)
    p, r, f, _ = precision_recall_fscore_support(label_train, y_pred)
    # cm = confusion_matrix(label_train, y_pred)
    log.append((e, loss.data[0], f[0], f[1]))
    if e % 10 == 0:
        print('Epoch %d: %f' % (e, loss))
print('DONE')



from matplotlib import pyplot as plt
epochs, losses, f0, f1 = zip(*log)
figure = plt.plot(epochs, losses, 'r-',epochs,f0,'b-',epochs,f1,'g-')
plt.show()
